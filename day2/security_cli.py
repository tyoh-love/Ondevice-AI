#!/usr/bin/env python3
"""
Command-line interface for AI Security System - Headless Video Analysis

Usage:
    python security_cli.py --video sample.mp4 --output-dir results/
    python security_cli.py -v video.mp4 -o results/ --confidence 0.7 --save-video
"""

import argparse
import sys
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

class HeadlessSecuritySystem:
    """AI 기반 스마트 보안 시스템 - CLI 버전 (UI 없음)"""
    
    def __init__(self, config: Dict = None, output_dir: str = "security_output"):
        """시스템 초기화"""
        print("🔒 AI 보안 시스템 초기화 중...")
        
        # 설정
        self.config = config or self.get_default_config()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 모델 초기화
        self.init_models()
        
        # 추적기 초기화
        self.tracker = sv.ByteTrack()
        
        # 데이터 저장소
        self.init_storage()
        
        print("✅ 시스템 초기화 완료!")
    
    def get_default_config(self) -> Dict:
        """기본 설정값"""
        return {
            'yolo_model': 'yolo11m.pt',
            'confidence_threshold': 0.5,
            'max_tracked_objects': 50,
            'suspicious_behaviors': {
                'loitering': {'duration': 60, 'area': 100},
                'running': {'speed_threshold': 5.0},
                'abandoned_object': {'duration': 300}
            }
        }
    
    def init_models(self):
        """AI 모델 초기화"""
        print("  🧠 YOLO 모델 로딩 중...")
        self.yolo = YOLO(self.config['yolo_model'])
        self.behavior_analyzer = BehaviorAnalyzer(self.config['suspicious_behaviors'])
        
    def init_storage(self):
        """데이터 저장 시스템 초기화"""
        # 추적 데이터
        self.track_history = defaultdict(lambda: {
            'positions': deque(maxlen=300),
            'timestamps': deque(maxlen=300),
            'class': None,
            'first_seen': None,
            'last_seen': None,
            'alerts': []
        })
        
        # 이벤트 로그
        self.event_log = []
        self.frame_count = 0
        self.alerts_count = defaultdict(int)
        
    def process_frame(self, frame: np.ndarray, frame_id: int, save_annotated: bool = False) -> Tuple[Optional[np.ndarray], List[Dict]]:
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
                    'time': current_time,
                    'frame_id': frame_id
                })
            
            history['last_seen'] = current_time
        
        # 4. 행동 분석
        suspicious_activities = self.behavior_analyzer.analyze(
            self.track_history, current_time
        )
        
        # 5. 알림 생성 및 이벤트 기록
        for activity in suspicious_activities:
            self.alerts_count[activity['type']] += 1
            activity['frame_id'] = frame_id
            events.append(activity)
            
            # 콘솔 알림
            print(f"🚨 ALERT: {activity['type']} (Frame {frame_id})")
            print(f"   Details: {activity.get('details', {}).get('message', 'N/A')}")
        
        # 6. 시각화 (필요한 경우)
        annotated_frame = None
        if save_annotated:
            annotated_frame = self.visualize_results(frame, tracks, suspicious_activities, frame_id)
        
        return annotated_frame, events
    
    def visualize_results(self, frame: np.ndarray, tracks: sv.Detections, 
                         activities: List[Dict], frame_id: int) -> np.ndarray:
        """결과 시각화 (어노테이션)"""
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
                annotated, label,
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # 상태 표시
        self.draw_status(annotated, len(tracks), activities, frame_id)
        
        return annotated
    
    def draw_status(self, frame: np.ndarray, num_objects: int, activities: List[Dict], frame_id: int):
        """상태 정보 표시"""
        h, w = frame.shape[:2]
        
        # 상단 정보 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        frame[:50] = cv2.addWeighted(overlay[:50], 0.7, frame[:50], 0.3, 0)
        
        # 프레임 번호
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 객체 수
        cv2.putText(frame, f"Objects: {num_objects}", (200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 경고
        if activities:
            alert_text = f"ALERT: {activities[0]['type']}"
            cv2.putText(frame, alert_text, (400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def analyze_video_file(self, video_path: str, save_frames: bool = False, 
                          save_video: bool = False, progress_interval: int = 30) -> Dict:
        """비디오 파일 분석 (헤드리스 모드)"""
        
        print(f"\n📹 비디오 분석 시작: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 비디오 속성
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   총 프레임: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   해상도: {width}x{height}")
        
        # 비디오 저장 설정
        video_writer = None
        if save_video:
            output_video_path = os.path.join(self.output_dir, "analyzed_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"   결과 비디오 저장: {output_video_path}")
        
        # 프레임별 분석
        frame_id = 0
        start_time = time.time()
        
        print("\n🔍 분석 진행 중...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            annotated_frame, events = self.process_frame(
                frame, frame_id, save_annotated=(save_frames or save_video)
            )
            
            # 이벤트 로깅
            for event in events:
                self.log_event(event)
            
            # 어노테이션된 프레임 저장
            if save_frames and annotated_frame is not None:
                frame_filename = f"frame_{frame_id:06d}.jpg"
                frame_path = os.path.join(self.output_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
            
            # 비디오 저장
            if save_video and video_writer and annotated_frame is not None:
                video_writer.write(annotated_frame)
            
            # 진행률 표시
            if frame_id % progress_interval == 0:
                progress = (frame_id / total_frames) * 100
                elapsed = time.time() - start_time
                fps_current = frame_id / elapsed if elapsed > 0 else 0
                print(f"   진행률: {progress:.1f}% ({frame_id}/{total_frames}) - {fps_current:.1f} FPS")
            
            frame_id += 1
            self.frame_count = frame_id
        
        # 정리
        cap.release()
        if video_writer:
            video_writer.release()
        
        analysis_time = time.time() - start_time
        
        print(f"\n✅ 분석 완료!")
        print(f"   총 처리 시간: {analysis_time:.2f}초")
        print(f"   평균 FPS: {frame_id / analysis_time:.2f}")
        
        # 결과 생성
        return self.generate_analysis_report()
    
    def log_event(self, event: Dict):
        """이벤트 로깅"""
        event['timestamp'] = datetime.now().isoformat()
        self.event_log.append(event)
    
    def convert_to_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_json_serializable(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            return self.convert_to_json_serializable(obj.__dict__)
        else:
            return obj

    def generate_analysis_report(self) -> Dict:
        """분석 리포트 생성"""
        print("\n📊 분석 리포트 생성 중...")
        
        # 전체 통계
        total_objects = len(self.track_history)
        
        # 클래스별 통계
        class_counts = defaultdict(int)
        for track_data in self.track_history.values():
            if track_data['class']:
                class_counts[track_data['class']] += 1
        
        # 이벤트 통계
        event_counts = defaultdict(int)
        for event in self.event_log:
            event_counts[event['type']] += 1
        
        # 리포트 생성
        report = {
            'analysis_summary': {
                'total_frames_processed': int(self.frame_count),
                'total_objects_tracked': int(total_objects),
                'total_events': len(self.event_log),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'object_statistics': {
                'by_class': dict(class_counts),
                'total_unique_objects': int(total_objects)
            },
            'event_statistics': {
                'by_type': dict(event_counts),
                'alerts_summary': dict(self.alerts_count)
            },
            'detailed_events': self.convert_to_json_serializable(self.event_log),
            'configuration': self.convert_to_json_serializable(self.config)
        }
        
        # Convert entire report to ensure JSON compatibility
        report = self.convert_to_json_serializable(report)
        
        # JSON 파일로 저장
        report_path = os.path.join(self.output_dir, 'analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"   📄 리포트 저장: {report_path}")
        
        return report
    
    def print_summary(self, report: Dict):
        """콘솔에 요약 출력"""
        print("\n" + "="*60)
        print("📋 AI 보안 시스템 - 분석 결과 요약")
        print("="*60)
        
        summary = report['analysis_summary']
        print(f"처리된 프레임 수: {summary['total_frames_processed']:,}")
        print(f"추적된 객체 수: {summary['total_objects_tracked']}")
        print(f"감지된 이벤트 수: {summary['total_events']}")
        
        print(f"\n📊 객체 클래스별 통계:")
        for class_name, count in sorted(report['object_statistics']['by_class'].items()):
            print(f"  - {class_name}: {count}개")
        
        print(f"\n🚨 이벤트 유형별 통계:")
        for event_type, count in sorted(report['event_statistics']['by_type'].items()):
            print(f"  - {event_type}: {count}회")
        
        print(f"\n⚠️  보안 알림 통계:")
        alerts = report['event_statistics']['alerts_summary']
        if alerts:
            for alert_type, count in sorted(alerts.items()):
                print(f"  - {alert_type}: {count}회")
        else:
            print("  - 의심스러운 활동이 감지되지 않았습니다 ✅")
        
        print("="*60)


class BehaviorAnalyzer:
    """행동 분석기 (원본과 동일)"""
    
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


def main():
    parser = argparse.ArgumentParser(
        description="AI Security System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python security_cli.py --video sample.mp4
  python security_cli.py -v video.mp4 -o results/ --confidence 0.7 --save-video
  python security_cli.py -v surveillance.mp4 -o analysis/ --save-frames --progress 60
        """
    )
    
    parser.add_argument(
        '-v', '--video',
        required=True,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='security_output',
        help='Output directory for results (default: security_output)'
    )
    
    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save annotated frames as images'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save annotated video'
    )
    
    parser.add_argument(
        '--progress',
        type=int,
        default=30,
        help='Progress update interval in frames (default: 30)'
    )
    
    parser.add_argument(
        '--model',
        default='yolo11m.pt',
        help='YOLO model to use (default: yolo11m.pt)'
    )
    
    args = parser.parse_args()
    
    print("🛡️  AI Security System - CLI Mode")
    print("="*50)
    
    try:
        # 설정 생성
        config = {
            'yolo_model': args.model,
            'confidence_threshold': args.confidence,
            'max_tracked_objects': 50,
            'suspicious_behaviors': {
                'loitering': {'duration': 60, 'area': 100},
                'running': {'speed_threshold': 5.0},
                'abandoned_object': {'duration': 300}
            }
        }
        
        # 시스템 초기화
        security_system = HeadlessSecuritySystem(config, args.output_dir)
        
        # 비디오 분석
        report = security_system.analyze_video_file(
            args.video,
            save_frames=args.save_frames,
            save_video=args.save_video,
            progress_interval=args.progress
        )
        
        # 결과 출력
        security_system.print_summary(report)
        
        print(f"\n📁 모든 결과가 '{args.output_dir}' 디렉토리에 저장되었습니다.")
        
    except KeyboardInterrupt:
        print("\n⚠️  분석이 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()