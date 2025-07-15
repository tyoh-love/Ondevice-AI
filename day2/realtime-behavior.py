# smart_security_system.py - 스마트 보안 시스템

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
from datetime import datetime
import json
import os
import threading
import queue
from typing import Dict, List, Tuple, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

class SmartSecuritySystem:
    """AI 기반 스마트 보안 시스템"""
    
    def __init__(self, config: Dict = None):
        """시스템 초기화"""
        print("🔒 스마트 보안 시스템을 초기화하고 있습니다...")
        
        # 설정
        self.config = config or self.get_default_config()
        
        # 모델 초기화
        self.init_models()
        
        # 추적기 초기화
        self.tracker = sv.ByteTrack()
        
        # 데이터 저장소
        self.init_storage()
        
        # 알림 시스템
        self.init_notification_system()
        
        print("✅ 시스템 초기화 완료!")
    
    def get_default_config(self) -> Dict:
        """기본 설정값"""
        return {
            'yolo_model': 'yolo11m.pt',
            'confidence_threshold': 0.5,
            'max_tracked_objects': 50,
            'alert_cooldown': 30,  # 초
            'recording_path': 'security_recordings',
            'suspicious_behaviors': {
                'loitering': {'duration': 60, 'area': 100},  # 60초 이상 머무르기
                'running': {'speed_threshold': 5.0},  # 빠른 움직임
                'intrusion': {'restricted_zones': []},  # 제한 구역 침입
                'crowding': {'max_people': 10},  # 과밀
                'abandoned_object': {'duration': 300}  # 5분 이상 방치된 물체
            },
            'notification': {
                'email': None,
                'webhook': None
            }
        }
    
    def init_models(self):
        """AI 모델 초기화"""
        print("  🧠 AI 모델을 로딩하고 있습니다...")
        
        # YOLO 모델
        self.yolo = YOLO(self.config['yolo_model'])
        
        # 행동 분류기 (간단한 규칙 기반)
        self.behavior_analyzer = BehaviorAnalyzer(self.config['suspicious_behaviors'])
        
    def init_storage(self):
        """데이터 저장 시스템 초기화"""
        self.storage_path = self.config['recording_path']
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 추적 데이터
        self.track_history = defaultdict(lambda: {
            'positions': deque(maxlen=300),  # 10초 @ 30fps
            'timestamps': deque(maxlen=300),
            'class': None,
            'first_seen': None,
            'last_seen': None,
            'alerts': []
        })
        
        # 이벤트 로그
        self.event_log = []
        
    def init_notification_system(self):
        """알림 시스템 초기화"""
        self.alert_queue = queue.Queue()
        self.last_alert_time = defaultdict(float)
        
        # 알림 처리 스레드
        self.notification_thread = threading.Thread(
            target=self.process_notifications,
            daemon=True
        )
        self.notification_thread.start()
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, List[Dict]]:
        """단일 프레임 처리"""
        
        # 1. 객체 탐지
        results = self.yolo(frame, conf=self.config['confidence_threshold'], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. 객체 추적
        tracks = self.tracker.update_with_detections(detections)
        
        # 3. 추적 데이터 업데이트
        current_time = time.time()
        events = []
        
        for i in range(len(tracks)):
            track_id = tracks.tracker_id[i]
            class_id = tracks.class_id[i]
            class_name = self.yolo.names[class_id]
            bbox = tracks.xyxy[i]
            
            # 중심점 계산
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # 히스토리 업데이트
            history = self.track_history[track_id]
            history['positions'].append(center)
            history['timestamps'].append(current_time)
            history['class'] = class_name
            
            if history['first_seen'] is None:
                history['first_seen'] = current_time
                events.append({
                    'type': 'new_object',
                    'track_id': track_id,
                    'class': class_name,
                    'time': current_time
                })
            
            history['last_seen'] = current_time
        
        # 4. 행동 분석
        suspicious_activities = self.behavior_analyzer.analyze(
            self.track_history, current_time
        )
        
        # 5. 알림 생성
        for activity in suspicious_activities:
            if self.should_send_alert(activity):
                self.alert_queue.put({
                    'activity': activity,
                    'frame': frame.copy(),
                    'time': current_time
                })
                events.append(activity)
        
        # 6. 시각화
        annotated_frame = self.visualize_results(
            frame, tracks, suspicious_activities
        )
        
        return annotated_frame, events
    
    def should_send_alert(self, activity: Dict) -> bool:
        """알림을 보낼지 결정"""
        alert_type = activity['type']
        current_time = time.time()
        
        # 쿨다운 확인
        if current_time - self.last_alert_time[alert_type] < self.config['alert_cooldown']:
            return False
        
        self.last_alert_time[alert_type] = current_time
        return True
    
    def visualize_results(self, frame: np.ndarray, tracks: sv.Detections, 
                         activities: List[Dict]) -> np.ndarray:
        """결과 시각화"""
        annotated = frame.copy()
        
        # 박스 그리기
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(scene=annotated, detections=tracks)
        
        # 추적 ID와 클래스 표시
        for i in range(len(tracks)):
            track_id = tracks.tracker_id[i]
            class_id = tracks.class_id[i]
            class_name = self.yolo.names[class_id]
            bbox = tracks.xyxy[i]
            
            # 라벨
            label = f"ID:{track_id} {class_name}"
            
            # 의심 행동이 있는 경우 강조
            is_suspicious = any(
                activity.get('track_id') == track_id 
                for activity in activities
            )
            
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            
            cv2.putText(
                annotated,
                label,
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # 상태 표시
        self.draw_status(annotated, len(tracks), activities)
        
        return annotated
    
    def draw_status(self, frame: np.ndarray, num_objects: int, activities: List[Dict]):
        """상태 정보 표시"""
        h, w = frame.shape[:2]
        
        # 상단 정보 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        frame[:50] = cv2.addWeighted(overlay[:50], 0.7, frame[:50], 0.3, 0)
        
        # 시간
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, current_time, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 객체 수
        cv2.putText(frame, f"Objects: {num_objects}", (300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 경고
        if activities:
            alert_text = f"ALERT: {activities[0]['type']}"
            cv2.putText(frame, alert_text, (500, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def process_notifications(self):
        """알림 처리 스레드"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1)
                self.send_notification(alert)
            except queue.Empty:
                continue
    
    def send_notification(self, alert: Dict):
        """알림 전송"""
        print(f"\n🚨 경고: {alert['activity']['type']}")
        print(f"   시간: {datetime.fromtimestamp(alert['time'])}")
        print(f"   세부사항: {alert['activity'].get('details', 'N/A')}")
        
        # 이메일 알림 (설정된 경우)
        if self.config['notification']['email']:
            self.send_email_alert(alert)
        
        # 웹훅 알림 (설정된 경우)
        if self.config['notification']['webhook']:
            self.send_webhook_alert(alert)
        
        # 스크린샷 저장
        self.save_alert_screenshot(alert)
    
    def save_alert_screenshot(self, alert: Dict):
        """경고 스크린샷 저장"""
        timestamp = datetime.fromtimestamp(alert['time']).strftime('%Y%m%d_%H%M%S')
        filename = f"{alert['activity']['type']}_{timestamp}.jpg"
        filepath = os.path.join(self.storage_path, filename)
        
        cv2.imwrite(filepath, alert['frame'])
        print(f"   📸 스크린샷 저장: {filepath}")
    
    def run_live_monitoring(self, source=0):
        """실시간 모니터링 실행"""
        print("\n🎥 실시간 모니터링을 시작합니다...")
        print("'q' 키를 눌러 종료")
        print("'s' 키를 눌러 스크린샷 저장")
        print("'r' 키를 눌러 녹화 시작/중지")
        
        cap = cv2.VideoCapture(source)
        
        # 비디오 속성
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 녹화 설정
        recording = False
        video_writer = None
        
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            annotated_frame, events = self.process_frame(frame, frame_id)
            
            # 이벤트 로깅
            for event in events:
                self.log_event(event)
            
            # 녹화
            if recording and video_writer:
                video_writer.write(annotated_frame)
            
            # 화면 표시
            cv2.imshow('Smart Security System', annotated_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(annotated_frame)
            elif key == ord('r'):
                if not recording:
                    # 녹화 시작
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    video_path = os.path.join(self.storage_path, f"recording_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    recording = True
                    print(f"🔴 녹화 시작: {video_path}")
                else:
                    # 녹화 중지
                    video_writer.release()
                    video_writer = None
                    recording = False
                    print("⏹️ 녹화 중지")
            
            frame_id += 1
        
        # 정리
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 최종 리포트
        self.generate_report()
    
    def log_event(self, event: Dict):
        """이벤트 로깅"""
        event['timestamp'] = datetime.now().isoformat()
        self.event_log.append(event)
        
        # 주기적으로 파일에 저장
        if len(self.event_log) % 100 == 0:
            self.save_event_log()
    
    def save_event_log(self):
        """이벤트 로그 저장"""
        log_path = os.path.join(self.storage_path, 'event_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.event_log, f, ensure_ascii=False, indent=2)
    
    def generate_report(self):
        """모니터링 리포트 생성"""
        print("\n📊 모니터링 리포트")
        print("=" * 50)
        
        # 전체 통계
        total_objects = len(self.track_history)
        print(f"총 추적 객체 수: {total_objects}")
        
        # 클래스별 통계
        class_counts = defaultdict(int)
        for track_data in self.track_history.values():
            if track_data['class']:
                class_counts[track_data['class']] += 1
        
        print("\n클래스별 객체 수:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  - {class_name}: {count}")
        
        # 경고 통계
        alert_counts = defaultdict(int)
        for event in self.event_log:
            if event.get('type') in ['loitering', 'running', 'intrusion']:
                alert_counts[event['type']] += 1
        
        print("\n경고 유형별 횟수:")
        for alert_type, count in sorted(alert_counts.items()):
            print(f"  - {alert_type}: {count}")
        
        print("=" * 50)


class BehaviorAnalyzer:
    """행동 분석기"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze(self, track_history: Dict, current_time: float) -> List[Dict]:
        """의심스러운 행동 분석"""
        suspicious_activities = []
        
        for track_id, history in track_history.items():
            if len(history['positions']) < 10:
                continue
            
            # 1. 배회 감지 (Loitering)
            loitering = self.detect_loitering(history, current_time)
            if loitering:
                suspicious_activities.append({
                    'type': 'loitering',
                    'track_id': track_id,
                    'details': loitering,
                    'time': current_time
                })
            
            # 2. 빠른 움직임 감지 (Running)
            running = self.detect_running(history)
            if running:
                suspicious_activities.append({
                    'type': 'running',
                    'track_id': track_id,
                    'details': running,
                    'time': current_time
                })
            
            # 3. 방치된 물체 감지
            if history['class'] in ['backpack', 'suitcase', 'handbag']:
                abandoned = self.detect_abandoned_object(history, current_time)
                if abandoned:
                    suspicious_activities.append({
                        'type': 'abandoned_object',
                        'track_id': track_id,
                        'details': abandoned,
                        'time': current_time
                    })
        
        return suspicious_activities
    
    def detect_loitering(self, history: Dict, current_time: float) -> Optional[Dict]:
        """배회 감지"""
        duration = current_time - history['first_seen']
        
        if duration < self.config['loitering']['duration']:
            return None
        
        # 이동 범위 계산
        positions = list(history['positions'])
        if len(positions) < 2:
            return None
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        area = x_range * y_range
        
        if area < self.config['loitering']['area']:
            return {
                'duration': duration,
                'area': area,
                'message': f"{duration:.1f}초 동안 작은 영역에 머물러 있음"
            }
        
        return None
    
    def detect_running(self, history: Dict) -> Optional[Dict]:
        """빠른 움직임 감지"""
        positions = list(history['positions'])
        timestamps = list(history['timestamps'])
        
        if len(positions) < 5:
            return None
        
        # 최근 5프레임의 속도 계산
        speeds = []
        for i in range(len(positions) - 5, len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            dt = timestamps[i+1] - timestamps[i]
            
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
                speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0
        
        if avg_speed > self.config['running']['speed_threshold']:
            return {
                'speed': avg_speed,
                'message': f"빠른 속도로 이동 중 ({avg_speed:.1f} pixels/s)"
            }
        
        return None
    
    def detect_abandoned_object(self, history: Dict, current_time: float) -> Optional[Dict]:
        """방치된 물체 감지"""
        # 마지막으로 본 시간 확인
        time_since_last_movement = current_time - history['last_seen']
        
        if time_since_last_movement > 1.0:  # 1초 이상 업데이트 없음
            # 정지 상태 확인
            positions = list(history['positions'])[-10:]
            if len(positions) < 10:
                return None
            
            # 위치 변화 계산
            position_std = np.std([p[0] for p in positions]) + np.std([p[1] for p in positions])
            
            if position_std < 5.0:  # 거의 움직임 없음
                stationary_duration = current_time - history['first_seen']
                
                if stationary_duration > self.config['abandoned_object']['duration']:
                    return {
                        'duration': stationary_duration,
                        'message': f"물체가 {stationary_duration:.1f}초 동안 방치됨"
                    }
        
        return None


# 웹 대시보드
def create_security_dashboard():
    """보안 시스템 웹 대시보드"""
    import gradio as gr
    import plotly.graph_objects as go
    
    security_system = SmartSecuritySystem()
    
    def process_video_file(video_file, confidence_threshold):
        """업로드된 비디오 파일 처리"""
        if video_file is None:
            return None, "비디오를 업로드해주세요"
        
        # 임시 설정 업데이트
        security_system.config['confidence_threshold'] = confidence_threshold
        
        cap = cv2.VideoCapture(video_file.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frames = []
        events = []
        frame_id = 0
        
        while len(frames) < 150:  # 최대 5초 처리
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, frame_events = security_system.process_frame(frame, frame_id)
            frames.append(annotated_frame)
            events.extend(frame_events)
            frame_id += 1
        
        cap.release()
        
        # 결과 비디오 생성
        output_path = "analyzed_video.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # 이벤트 요약
        event_summary = f"총 {len(events)}개의 이벤트가 감지되었습니다.\n"
        event_types = defaultdict(int)
        for event in events:
            event_types[event['type']] += 1
        
        for event_type, count in event_types.items():
            event_summary += f"- {event_type}: {count}회\n"
        
        return output_path, event_summary
    
    def generate_statistics():
        """통계 그래프 생성"""
        # 시간대별 객체 수
        hours = list(range(24))
        object_counts = np.random.poisson(10, 24)  # 예시 데이터
        
        fig1 = go.Figure(data=go.Bar(x=hours, y=object_counts))
        fig1.update_layout(
            title="시간대별 객체 탐지 수",
            xaxis_title="시간",
            yaxis_title="객체 수"
        )
        
        # 클래스별 분포
        classes = ['person', 'car', 'bicycle', 'motorcycle', 'truck']
        class_counts = [45, 30, 15, 8, 12]
        
        fig2 = go.Figure(data=go.Pie(labels=classes, values=class_counts))
        fig2.update_layout(title="객체 클래스 분포")
        
        return fig1, fig2
    
    # Gradio 인터페이스
    with gr.Blocks(title="AI 보안 시스템", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔒 AI 기반 스마트 보안 시스템
        
        실시간 객체 탐지, 추적, 그리고 이상 행동 감지
        """)
        
        with gr.Tab("📹 비디오 분석"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="비디오 업로드")
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.1,
                        label="탐지 신뢰도 임계값"
                    )
                    analyze_btn = gr.Button("분석 시작", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="분석 결과")
                    event_summary = gr.Textbox(label="이벤트 요약", lines=5)
            
            analyze_btn.click(
                fn=process_video_file,
                inputs=[video_input, confidence_slider],
                outputs=[video_output, event_summary]
            )
        
        with gr.Tab("📊 통계 대시보드"):
            with gr.Row():
                hourly_chart = gr.Plot(label="시간대별 통계")
                class_chart = gr.Plot(label="객체 클래스 분포")
            
            refresh_btn = gr.Button("통계 새로고침")
            refresh_btn.click(
                fn=generate_statistics,
                outputs=[hourly_chart, class_chart]
            )
        
        with gr.Tab("⚙️ 설정"):
            gr.Markdown("""
            ### 시스템 설정
            
            - **알림 이메일**: security@example.com
            - **녹화 저장 경로**: /security_recordings
            - **최대 추적 객체 수**: 50
            - **알림 쿨다운**: 30초
            
            ### 의심 행동 설정
            
            - **배회**: 60초 이상 같은 장소에 머무르기
            - **빠른 움직임**: 5 pixels/s 이상의 속도
            - **방치된 물체**: 5분 이상 움직임 없음
            """)
        
        # 예시 추가
        gr.Examples(
            examples=[
                ["example_security_1.mp4", 0.5],
                ["example_security_2.mp4", 0.7]
            ],
            inputs=[video_input, confidence_slider]
        )
    
    return demo

# 시스템 실행
def run_security_system():
    """보안 시스템 실행"""
    print("""
    ╔══════════════════════════════════════╗
    ║      🔒 AI 스마트 보안 시스템 🔒       ║
    ║                                      ║
    ║   실시간 감시와 이상 행동 탐지        ║
    ╚══════════════════════════════════════╝
    """)
    
    print("\n실행 모드를 선택하세요:")
    print("1. 실시간 카메라 모니터링")
    print("2. 비디오 파일 분석")
    print("3. 웹 대시보드")
    
    choice = input("\n선택 (1-3): ")
    
    if choice == "1":
        system = SmartSecuritySystem()
        system.run_live_monitoring(source=0)  # 웹캠
        
    elif choice == "2":
        video_path = input("비디오 파일 경로: ")
        system = SmartSecuritySystem()
        system.run_live_monitoring(source=video_path)
        
    elif choice == "3":
        dashboard = create_security_dashboard()
        dashboard.launch(share=True)
    
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    run_security_system()