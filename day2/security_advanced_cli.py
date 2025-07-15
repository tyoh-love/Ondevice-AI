#!/usr/bin/env python3
"""
Command-line version of Advanced Security System for WSL/headless environments
No GUI dependencies - pure processing and file output
"""

import numpy as np
import time
import os
import sys
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Prevent any Qt/GUI initialization
os.environ['OPENCV_HEADLESS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Now import cv2 after setting headless mode
import cv2

# Set cv2 to not use any GUI backend
cv2.setUseOptimized(True)

# Import other required modules
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
    print("Warning: face_recognition not installed. Face detection will be disabled.")

import supervision as sv
from sklearn.cluster import DBSCAN
import networkx as nx

# Import security system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("realtime_behavior", "realtime-behavior.py")
    realtime_behavior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realtime_behavior)
    SmartSecuritySystem = realtime_behavior.SmartSecuritySystem
    BehaviorAnalyzer = realtime_behavior.BehaviorAnalyzer
except Exception as e:
    print(f"Error importing security system: {e}")
    sys.exit(1)


class HeadlessAdvancedSecurity:
    """Advanced security features for headless operation"""
    
    def __init__(self, output_dir="advanced_security_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.security_system = SmartSecuritySystem()
        self.face_encodings = {}
        self.analysis_data = {
            'faces_detected': 0,
            'crowd_moments': [],
            'path_predictions': {},
            'zone_violations': [],
            'network_connections': defaultdict(set)
        }
    
    def process_video(self, video_path: str, save_annotated: bool = True):
        """Process video file without any GUI operations"""
        
        print(f"\n🎬 Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if saving annotated video
        out_writer = None
        if save_annotated:
            output_path = os.path.join(self.output_dir, "analyzed_advanced.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving annotated video to: {output_path}")
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Basic security analysis
            annotated_frame, events = self.security_system.process_frame(frame, frame_count)
            
            # Advanced features (if available)
            if HAS_FACE_RECOGNITION and frame_count % 10 == 0:  # Face recognition every 10 frames
                self.analyze_faces(frame, annotated_frame)
            
            # Crowd analysis
            detections = self.get_detections(frame)
            if len(detections) > 5:  # Crowd threshold
                self.analysis_data['crowd_moments'].append({
                    'frame': frame_count,
                    'count': len(detections),
                    'timestamp': frame_count / fps
                })
            
            # Save annotated frame
            if out_writer:
                out_writer.write(annotated_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                elapsed = time.time() - start_time
                eta = (elapsed / (frame_count + 1)) * (total_frames - frame_count) if frame_count > 0 else 0
                print(f"\rProgress: {progress:.1f}% ({frame_count}/{total_frames}) - ETA: {eta:.0f}s", end='', flush=True)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        
        # Analysis complete
        analysis_time = time.time() - start_time
        print(f"\n\n✅ Analysis complete!")
        print(f"   Processed {frame_count} frames in {analysis_time:.1f} seconds")
        print(f"   Average FPS: {frame_count / analysis_time:.1f}")
        
        # Generate report
        self.generate_report(frame_count, fps)
    
    def get_detections(self, frame: np.ndarray) -> sv.Detections:
        """Get object detections from frame"""
        try:
            results = self.security_system.yolo(frame, conf=0.5, verbose=False)[0]
            return sv.Detections.from_ultralytics(results)
        except:
            return sv.Detections.empty()
    
    def analyze_faces(self, frame: np.ndarray, annotated_frame: np.ndarray):
        """Analyze faces in frame (if face_recognition is available)"""
        if not HAS_FACE_RECOGNITION:
            return
        
        try:
            # Find faces
            face_locations = face_recognition.face_locations(frame, model='hog')
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), (255, 0, 255), 2)
                cv2.putText(annotated_frame, "Face", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                self.analysis_data['faces_detected'] += 1
        except Exception as e:
            pass  # Silently fail face detection
    
    def generate_report(self, total_frames: int, fps: float):
        """Generate analysis report"""
        print("\n📊 Generating analysis report...")
        
        report = {
            'summary': {
                'total_frames': total_frames,
                'duration_seconds': total_frames / fps,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'security_analysis': {
                'objects_tracked': len(self.security_system.track_history),
                'events_detected': len(self.security_system.event_log),
                'alerts': dict(self.security_system.alerts_count) if hasattr(self.security_system, 'alerts_count') else {}
            },
            'advanced_analysis': {
                'faces_detected': self.analysis_data['faces_detected'],
                'crowd_moments': len(self.analysis_data['crowd_moments']),
                'max_crowd_size': max([m['count'] for m in self.analysis_data['crowd_moments']], default=0)
            },
            'detailed_data': {
                'crowd_timeline': self.analysis_data['crowd_moments'][:10],  # First 10 crowd moments
                'event_log': self.security_system.event_log[:20]  # First 20 events
            }
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'advanced_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("📋 Analysis Summary")
        print("="*60)
        print(f"Duration: {report['summary']['duration_seconds']:.1f} seconds")
        print(f"Objects tracked: {report['security_analysis']['objects_tracked']}")
        print(f"Security events: {report['security_analysis']['events_detected']}")
        print(f"Faces detected: {report['advanced_analysis']['faces_detected']}")
        print(f"Crowd moments: {report['advanced_analysis']['crowd_moments']}")
        if report['security_analysis']['alerts']:
            print("\n⚠️  Security Alerts:")
            for alert_type, count in report['security_analysis']['alerts'].items():
                print(f"  - {alert_type}: {count}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Security System - Headless CLI Version",
        epilog="""
Examples:
  python security_advanced_cli.py -v video.mp4
  python security_advanced_cli.py -v video.mp4 -o results/
  python security_advanced_cli.py -v video.mp4 --no-output-video
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-v', '--video', required=True, help='Video file path')
    parser.add_argument('-o', '--output-dir', default='advanced_security_output', 
                       help='Output directory (default: advanced_security_output)')
    parser.add_argument('--no-output-video', action='store_true',
                       help='Skip saving annotated video (faster processing)')
    
    args = parser.parse_args()
    
    print("🛡️  Advanced Security System - CLI Mode")
    print("="*50)
    print("Running in headless mode (no GUI)")
    
    try:
        # Create processor
        processor = HeadlessAdvancedSecurity(args.output_dir)
        
        # Process video
        processor.process_video(
            args.video,
            save_annotated=not args.no_output_video
        )
        
        print(f"\n✅ All results saved to: {args.output_dir}/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()