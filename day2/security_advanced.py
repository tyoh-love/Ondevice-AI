
import numpy as np
import time
import face_recognition
import pickle
import supervision as sv
from sklearn.cluster import DBSCAN
import networkx as nx
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# Import SmartSecuritySystem from the main security module
# Note: Python module names can't have hyphens, so we need to import it differently
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for headless operation BEFORE importing cv2
# This prevents Qt platform errors in WSL/SSH environments
if 'WSL_DISTRO_NAME' in os.environ or 'SSH_CLIENT' in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    print("WSL/SSH environment detected - running in compatibility mode")

import cv2

try:
    # Import the module with hyphen in filename
    import importlib.util
    spec = importlib.util.spec_from_file_location("realtime_behavior", "realtime-behavior.py")
    realtime_behavior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realtime_behavior)
    SmartSecuritySystem = realtime_behavior.SmartSecuritySystem
except Exception as e:
    print(f"Warning: Could not import SmartSecuritySystem: {e}")
    # If running as standalone, define a mock class
    class SmartSecuritySystem:
        pass

class AdvancedSecurityFeatures:
    """고급 보안 기능 모음"""
    
    def __init__(self):
        self.known_faces = {}
        self.load_known_faces()
    
    def load_known_faces(self):
        """알려진 얼굴 데이터 로드"""
        try:
            with open('known_faces.pkl', 'rb') as f:
                self.known_faces = pickle.load(f)
        except FileNotFoundError:
            print("알려진 얼굴 데이터가 없습니다.")
    
    def face_recognition_analysis(self, frame: np.ndarray) -> List[Dict]:
        """얼굴 인식 분석"""
        # 얼굴 찾기
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 알려진 얼굴과 비교
            matches = face_recognition.compare_faces(
                list(self.known_faces.values()),
                face_encoding
            )
            
            name = "Unknown"
            
            if True in matches:
                match_index = matches.index(True)
                name = list(self.known_faces.keys())[match_index]
            
            results.append({
                'name': name,
                'location': (left, top, right, bottom),
                'authorized': name != "Unknown"
            })
        
        return results
    
    def crowd_analysis(self, detections: sv.Detections) -> Dict:
        """군중 분석"""
        # 사람만 필터링
        person_indices = [i for i, class_id in enumerate(detections.class_id) 
                         if class_id == 0]  # 0은 'person' 클래스
        
        if len(person_indices) < 2:
            return {'density': 'low', 'clusters': 0}
        
        # 위치 추출
        positions = []
        for i in person_indices:
            bbox = detections.xyxy[i]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            positions.append(center)
        
        positions = np.array(positions)
        
        # DBSCAN 클러스터링
        clustering = DBSCAN(eps=100, min_samples=3).fit(positions)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        # 밀도 계산
        total_people = len(person_indices)
        if total_people < 5:
            density = 'low'
        elif total_people < 15:
            density = 'medium'
        else:
            density = 'high'
        
        return {
            'density': density,
            'clusters': n_clusters,
            'total_people': total_people,
            'positions': positions,
            'labels': clustering.labels_
        }
    
    def path_prediction(self, track_history: Dict, track_id: int) -> Optional[np.ndarray]:
        """경로 예측"""
        if track_id not in track_history:
            return None
        
        positions = list(track_history[track_id]['positions'])
        
        if len(positions) < 10:
            return None
        
        # 최근 10개 위치
        recent_positions = np.array(positions[-10:])
        
        # 간단한 선형 예측
        x_positions = recent_positions[:, 0]
        y_positions = recent_positions[:, 1]
        
        # 선형 회귀
        t = np.arange(len(x_positions))
        x_coef = np.polyfit(t, x_positions, 1)
        y_coef = np.polyfit(t, y_positions, 1)
        
        # 다음 5프레임 예측
        future_t = np.arange(len(x_positions), len(x_positions) + 5)
        future_x = np.polyval(x_coef, future_t)
        future_y = np.polyval(y_coef, future_t)
        
        future_positions = np.column_stack((future_x, future_y))
        
        return future_positions
    
    def zone_monitoring(self, frame_shape: Tuple[int, int]) -> Dict:
        """구역별 모니터링 설정"""
        height, width = frame_shape
        
        zones = {
            'entrance': {
                'polygon': np.array([
                    [0, height * 0.7],
                    [width * 0.3, height * 0.7],
                    [width * 0.3, height],
                    [0, height]
                ], dtype=np.int32),
                'type': 'entrance',
                'max_loitering_time': 30
            },
            'restricted': {
                'polygon': np.array([
                    [width * 0.7, 0],
                    [width, 0],
                    [width, height * 0.3],
                    [width * 0.7, height * 0.3]
                ], dtype=np.int32),
                'type': 'restricted',
                'authorized_only': True
            },
            'exit': {
                'polygon': np.array([
                    [width * 0.7, height * 0.7],
                    [width, height * 0.7],
                    [width, height],
                    [width * 0.7, height]
                ], dtype=np.int32),
                'type': 'exit',
                'direction': 'out'
            }
        }
        
        return zones
    
    def check_zone_violations(self, position: Tuple[float, float], 
                            zones: Dict, is_authorized: bool = False) -> List[str]:
        """구역 위반 확인"""
        violations = []
        
        point = np.array(position, dtype=np.float32)
        
        for zone_name, zone_info in zones.items():
            # 점이 다각형 내부에 있는지 확인
            result = cv2.pointPolygonTest(zone_info['polygon'], point, False)
            
            if result >= 0:  # 내부 또는 경계
                if zone_info['type'] == 'restricted' and not is_authorized:
                    violations.append(f"Unauthorized access to {zone_name}")
                
        return violations
    
    def network_analysis(self, track_history: Dict, time_window: float = 60) -> nx.Graph:
        """객체 간 네트워크 분석"""
        G = nx.Graph()
        
        current_time = time.time()
        
        # 활성 트랙만 선택
        active_tracks = {
            track_id: data for track_id, data in track_history.items()
            if current_time - data['last_seen'] < time_window
        }
        
        # 노드 추가
        for track_id, data in active_tracks.items():
            G.add_node(track_id, 
                      object_class=data['class'],
                      duration=current_time - data['first_seen'])
        
        # 엣지 추가 (근접성 기반)
        track_ids = list(active_tracks.keys())
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1, id2 = track_ids[i], track_ids[j]
                
                # 최근 위치
                if active_tracks[id1]['positions'] and active_tracks[id2]['positions']:
                    pos1 = active_tracks[id1]['positions'][-1]
                    pos2 = active_tracks[id2]['positions'][-1]
                    
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if distance < 100:  # 근접 임계값
                        G.add_edge(id1, id2, weight=1/distance)
        
        return G
    
    def generate_heatmap(self, track_history: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """움직임 히트맵 생성"""
        height, width = frame_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for track_data in track_history.values():
            for position in track_data['positions']:
                x, y = int(position[0]), int(position[1])
                
                # 가우시안 커널 적용
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(heatmap, (x, y), 20, 1, -1)
        
        # 정규화
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # 컬러맵 적용
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored

# 실시간 분석 대시보드
class RealtimeAnalyticsDashboard:
    """실시간 분석 대시보드"""
    
    def __init__(self, security_system: SmartSecuritySystem):
        self.security_system = security_system
        self.advanced_features = AdvancedSecurityFeatures()
        
    def create_dashboard(self):
        """대시보드 생성"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import dash_bootstrap_components as dbc
        
        # Dash 앱 초기화
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🔒 실시간 보안 분석 대시보드", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='live-video-feed'),
                    dcc.Interval(id='video-update', interval=100)  # 100ms마다 업데이트
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("실시간 통계"),
                        dbc.CardBody([
                            html.H4(id='total-objects', children="0"),
                            html.P("총 객체 수"),
                            html.Hr(),
                            html.H4(id='alert-count', children="0"),
                            html.P("경고 횟수"),
                            html.Hr(),
                            html.H4(id='crowd-density', children="Low"),
                            html.P("군중 밀도")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='heatmap')
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id='network-graph')
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='timeline-chart'),
                    dcc.Interval(id='chart-update', interval=1000)  # 1초마다 업데이트
                ])
            ], className="mt-4")
        ], fluid=True)
        
        # 콜백 함수들
        @app.callback(
            [Output('total-objects', 'children'),
             Output('alert-count', 'children'),
             Output('crowd-density', 'children')],
            [Input('video-update', 'n_intervals')]
        )
        def update_stats(n):
            """통계 업데이트"""
            total_objects = len(self.security_system.track_history)
            alert_count = len([e for e in self.security_system.event_log 
                             if e.get('type') in ['loitering', 'running']])
            
            # 군중 밀도 계산 (간단한 예시)
            active_objects = sum(1 for track in self.security_system.track_history.values()
                               if time.time() - track['last_seen'] < 5)
            
            if active_objects < 5:
                density = "Low"
            elif active_objects < 15:
                density = "Medium"
            else:
                density = "High"
            
            return str(total_objects), str(alert_count), density
        
        @app.callback(
            Output('timeline-chart', 'figure'),
            [Input('chart-update', 'n_intervals')]
        )
        def update_timeline(n):
            """타임라인 차트 업데이트"""
            # 최근 60초 데이터
            current_time = time.time()
            time_bins = defaultdict(int)
            
            for event in self.security_system.event_log:
                if 'time' in event:
                    time_diff = current_time - event['time']
                    if time_diff < 60:
                        bin_index = int(time_diff / 5)  # 5초 단위
                        time_bins[bin_index] += 1
            
            x = list(range(12))
            y = [time_bins.get(11-i, 0) for i in x]
            
            fig = go.Figure(data=go.Bar(x=x, y=y))
            fig.update_layout(
                title="최근 60초 이벤트 타임라인",
                xaxis_title="시간 (5초 단위)",
                yaxis_title="이벤트 수",
                xaxis=dict(ticktext=[f"{i*5}s" for i in range(12)],
                          tickvals=list(range(12)))
            )
            
            return fig
        
        return app

# 전체 시스템 통합
def run_advanced_security_system(video_source=None, headless=False):
    """고급 보안 시스템 실행
    
    Args:
        video_source: 비디오 파일 경로 또는 카메라 인덱스 (기본값: None으로 사용자 입력 받음)
        headless: True면 화면 표시 없이 실행 (WSL 환경용)
    """
    
    print("""
    ╔══════════════════════════════════════════╗
    ║    🔒 고급 AI 보안 시스템 v2.0 🔒         ║
    ║                                          ║
    ║  • 얼굴 인식        • 군중 분석            ║
    ║  • 경로 예측        • 구역 모니터링         ║
    ║  • 네트워크 분석    • 히트맵 생성           ║
    ╚══════════════════════════════════════════╝
    """)
    
    # 비디오 소스 선택
    if video_source is None:
        print("\n실행 모드를 선택하세요:")
        print("1. 비디오 파일 분석")
        print("2. 웹캠 실시간 분석 (WSL에서는 작동하지 않을 수 있음)")
        
        choice = input("\n선택 (1-2): ")
        
        if choice == "1":
            video_source = input("비디오 파일 경로를 입력하세요: ")
            if not os.path.exists(video_source):
                print(f"오류: 파일을 찾을 수 없습니다 - {video_source}")
                return
        elif choice == "2":
            video_source = 0
            print("웹캠을 사용합니다. WSL 환경에서는 작동하지 않을 수 있습니다.")
        else:
            print("잘못된 선택입니다.")
            return
    
    security_system = SmartSecuritySystem()
    advanced_features = AdvancedSecurityFeatures()
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"오류: 비디오 소스를 열 수 없습니다 - {video_source}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if isinstance(video_source, str):
        print(f"\n비디오 파일 정보:")
        print(f"  경로: {video_source}")
        print(f"  총 프레임: {total_frames}")
        print(f"  FPS: {fps}")
    
    # 결과 저장을 위한 VideoWriter (headless 모드)
    out_writer = None
    if headless and isinstance(video_source, str):
        output_path = video_source.rsplit('.', 1)[0] + '_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"  분석 결과 저장: {output_path}")
    
    frame_count = 0
    print("\n분석 중... (ESC 또는 'q' 키로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 기본 처리
        annotated_frame, events = security_system.process_frame(frame, frame_count)
        
        # 얼굴 인식
        try:
            faces = advanced_features.face_recognition_analysis(frame)
            for face in faces:
                x1, y1, x2, y2 = face['location']
                color = (0, 255, 0) if face['authorized'] else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, face['name'], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            # 얼굴 인식 실패 시 계속 진행
            pass
        
        # 히트맵 오버레이
        if len(security_system.track_history) > 0:
            heatmap = advanced_features.generate_heatmap(
                security_system.track_history, 
                frame.shape[:2]
            )
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap, 0.3, 0)
        
        # 진행률 표시
        if isinstance(video_source, str) and frame_count % 30 == 0:
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"\r진행률: {progress:.1f}% ({frame_count}/{total_frames})", end='', flush=True)
        
        # 결과 저장 (headless 모드)
        if out_writer:
            out_writer.write(annotated_frame)
        
        # 화면 표시 (headless가 아닌 경우)
        if not headless:
            cv2.imshow('Advanced Security System', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                break
        
        frame_count += 1
    
    print(f"\n\n분석 완료! 총 {frame_count} 프레임 처리됨.")
    
    # 정리
    cap.release()
    if out_writer:
        out_writer.release()
    if not headless:
        cv2.destroyAllWindows()
    
    # 최종 통계 출력
    print("\n📊 분석 통계:")
    print(f"  추적된 객체 수: {len(security_system.track_history)}")
    print(f"  감지된 이벤트 수: {len(security_system.event_log)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="고급 AI 보안 시스템",
        epilog="""
사용 예시:
  일반 실행: python security_advanced.py
  비디오 파일: python security_advanced.py -v video.mp4
  WSL/헤드리스: python security_advanced.py -v video.mp4 --headless
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--video', help='비디오 파일 경로')
    parser.add_argument('--headless', action='store_true', 
                       help='화면 표시 없이 실행 (WSL/서버 환경용)')
    
    args = parser.parse_args()
    
    # WSL 환경에서 headless 모드 자동 권장
    if 'WSL_DISTRO_NAME' in os.environ and not args.headless and args.video:
        print("\n⚠️  WSL 환경이 감지되었습니다.")
        print("GUI 표시 문제를 피하려면 --headless 옵션을 사용하는 것을 권장합니다:")
        print(f"python {sys.argv[0]} -v {args.video} --headless\n")
        
        response = input("계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("--headless 옵션과 함께 다시 실행해주세요.")
            sys.exit(0)
    
    try:
        run_advanced_security_system(video_source=args.video, headless=args.headless)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        if "qt.qpa.plugin" in str(e).lower() or "xcb" in str(e).lower():
            print("\n💡 해결 방법: --headless 옵션을 사용하세요")
            print(f"   python {sys.argv[0]} -v {args.video} --headless")