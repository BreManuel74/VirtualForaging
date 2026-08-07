"""
A script to record video from a ThorCam camera using Micro-Manager and OpenCV.
The script captures frames at a specified FPS, saves them to an AVI file using the MJPEG codec,
and logs the timestamp and frame number to a text file. It supports hardware-triggered acquisition.
Original Author: Brenna Manuel
"""
import os
import cv2
import numpy as np
import pymmcore_plus
import time
import threading
from pathlib import Path
from global_stopwatch import Stopwatch

def auto_adjust_exposure(mmc, camera_device, target_mean_8bit=200, tolerance_8bit=10, max_iterations=15, max_exposure_ms=None):
    """
    Automatically adjust camera exposure to achieve optimal brightness.
    
    Args:
        mmc: Micro-Manager core instance
        camera_device: Name of the camera device
        target_mean_8bit: Target mean pixel value in 8-bit space (0-255, typically 100-140 is good)
        tolerance_8bit: Acceptable deviation from target in 8-bit space
        max_iterations: Maximum adjustment iterations
    """
    # Get bit depth and calculate target in native bit depth
    bit_depth = mmc.getImageBitDepth()
    max_value = (2 ** bit_depth) - 1
    
    # Scale target to native bit depth
    target_mean = (target_mean_8bit / 255.0) * max_value
    tolerance = (tolerance_8bit / 255.0) * max_value
    
    print(f"Camera bit depth: {bit_depth}-bit (max value: {max_value})")
    print(f"Auto-adjusting exposure for optimal brightness (target: {target_mean_8bit} in 8-bit = {target_mean:.0f} in {bit_depth}-bit)...")
    
    # Get current exposure and valid range
    current_exposure = float(mmc.getProperty(camera_device, "Exposure"))
    
    # Try to get exposure limits (if available)
    try:
        min_exposure = float(mmc.getPropertyLowerLimit(camera_device, "Exposure"))
        hw_max_exposure = float(mmc.getPropertyUpperLimit(camera_device, "Exposure"))
    except:
        min_exposure = 0.1
        hw_max_exposure = None

    # Caller-supplied cap takes precedence; hardware max is only used as fallback.
    # Hardware-reported upper limits can be unreliable (e.g. change with trigger mode).
    if max_exposure_ms is not None:
        max_exposure = max_exposure_ms
    elif hw_max_exposure is not None:
        max_exposure = hw_max_exposure
    else:
        max_exposure = 100.0

    # Clamp current exposure into the valid range before snapping any calibration frames
    clamped = max(min_exposure, min(current_exposure, max_exposure))
    if clamped != current_exposure:
        current_exposure = clamped
        mmc.setProperty(camera_device, "Exposure", current_exposure)
        time.sleep(0.1)

    print(f"Initial exposure: {current_exposure} ms (range: {min_exposure}-{max_exposure} ms)")
    print(f"Target brightness: {target_mean:.0f} / {max_value} ({target_mean_8bit} / 255 in 8-bit)")

    # Temporarily switch to internal trigger so snapImage() gets real frames.
    # In hardware-trigger mode snapImage() returns a black frame, causing the
    # loop to max out exposure and produce an overexposed recording.
    original_trigger_mode = None
    try:
        original_trigger_mode = mmc.getProperty(camera_device, "TriggerMode")
        if original_trigger_mode.lower() not in ("internal", "software"):
            mmc.setProperty(camera_device, "TriggerMode", "Internal")
            print(f"Switched trigger mode from '{original_trigger_mode}' to 'Internal' for calibration")
    except Exception:
        pass  # Camera may not expose TriggerMode property; proceed anyway

    try:
        for iteration in range(max_iterations):
            # Capture a test frame
            mmc.snapImage()
            image = mmc.getImage()

            # Calculate mean brightness
            mean_brightness = np.mean(image)

            print(f"Iteration {iteration + 1}: Exposure={current_exposure:.2f} ms, Mean brightness={mean_brightness:.1f}")

            # Guard: near-zero mean means frames aren't being captured (trigger issue).
            if mean_brightness < max_value * 0.005:
                print(f"⚠ Mean brightness is near zero — snapImage may not be receiving a trigger. "
                      f"Skipping auto-exposure adjustment.")
                return current_exposure

            # Check if we're within tolerance
            if abs(mean_brightness - target_mean) <= tolerance:
                print(f"✓ Optimal exposure found: {current_exposure:.2f} ms (brightness: {mean_brightness:.1f})")
                return current_exposure

            # Adjust exposure based on brightness
            if mean_brightness < target_mean - tolerance:
                # Image too dark, increase exposure
                adjustment_factor = min((target_mean / mean_brightness), 2.0)  # Cap at 2x increase per step
                new_exposure = current_exposure * adjustment_factor
            else:
                # Image too bright, decrease exposure
                adjustment_factor = max((target_mean / mean_brightness), 0.5)  # Cap at 0.5x decrease per step
                new_exposure = current_exposure * adjustment_factor

            # Clamp to valid range
            new_exposure = max(min_exposure, min(new_exposure, max_exposure))

            # Check if we're stuck at limits
            if new_exposure == current_exposure:
                print(f"⚠ Exposure at limit ({current_exposure:.2f} ms), cannot improve further")
                return current_exposure

            # Apply new exposure
            current_exposure = new_exposure
            mmc.setProperty(camera_device, "Exposure", current_exposure)
            time.sleep(0.1)  # Allow camera to adjust

        print(f"⚠ Max iterations reached. Final exposure: {current_exposure:.2f} ms (brightness: {mean_brightness:.1f})")
        return current_exposure
    finally:
        # Always restore the original trigger mode before returning
        if original_trigger_mode is not None:
            try:
                mmc.setProperty(camera_device, "TriggerMode", original_trigger_mode)
                print(f"Restored trigger mode to '{original_trigger_mode}'")
            except Exception:
                pass

def main():
    global_stopwatch = Stopwatch()
    
    # Use the start time from the main process if available
    start_time_str = os.environ.get("STOPWATCH_START_TIME")
    if start_time_str:
        global_stopwatch.start_time = float(start_time_str)
        global_stopwatch.running = True
        print(f"Synchronized with main process stopwatch (start time: {global_stopwatch.start_time})")
    else:
        global_stopwatch.start()
        print("Warning: No shared stopwatch time found, starting independent stopwatch")

    camera_device = "ThorCam"
    video_dir = os.environ.get("OUTPUT_DIR")
    fps = 20
    stop_file = "stop_recording.flag"
    
    # MJPEG codec is used instead of H264 for reliability
    # No external DLL dependencies needed

    # Initialize the Micro-Manager core
    mmc = pymmcore_plus.CMMCorePlus()
    mmc.loadSystemConfiguration(r"C:\Program Files\Micro-Manager-2.0\thorcam.cfg")
    mmc.setCameraDevice(camera_device)
    
    # Cap exposure to 90% of the frame period so hardware-triggered acquisition stays in sync
    auto_adjust_exposure(mmc, camera_device, target_mean_8bit=60, tolerance_8bit=10,
                         max_exposure_ms=(1000.0 / fps) * 0.9)
    
    # Detect bit depth for fast conversion to 8-bit using bit shift
    bit_depth = mmc.getImageBitDepth()
    
    # Use bit shift for fast conversion (integer operation, not float division)
    if bit_depth > 8:
        bit_shift = bit_depth - 8
        print(f"Camera: {bit_depth}-bit, converting to 8-bit using right shift by {bit_shift} bits")
    else:
        bit_shift = 0
        print(f"Camera: {bit_depth}-bit, no conversion needed")
    
    #print(mmc.getDevicePropertyNames(camera_device))

    # Video output settings
    os.makedirs(video_dir, exist_ok=True)
    out_filename = os.path.join(video_dir, f"{int(time.time())}pupil_cam.avi")
    frame_width = int(mmc.getImageWidth())
    frame_height = int(mmc.getImageHeight())
    
    # Use MJPEG - works reliably, no external dependencies, good quality
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Prepare text log file
    log_path = os.path.join(video_dir, f"{int(time.time())}_frame_log.txt")
    log_file = open(log_path, "w")
    log_file.write("time_seconds\tframe_number\n")

    # Start hardware-triggered sequence acquisition
    num_frames = 0  # 0 means indefinite acquisition until stopped manually

    # Shared display state — writer does a single reference assignment, display thread reads it
    latest_frame = [None]  # list so the thread closure can mutate it
    stop_display = threading.Event()

    def _display_worker():
        cv2.namedWindow("Live View", cv2.WINDOW_NORMAL)
        while not stop_display.is_set():
            frame = latest_frame[0]
            if frame is not None:
                cv2.imshow("Live View", frame)
                cv2.waitKey(1)
            time.sleep(0.1)  # ~10 fps display rate
        cv2.destroyWindow("Live View")

    display_thread = threading.Thread(target=_display_worker, daemon=True)
    display_thread.start()

    try:
        mmc.startSequenceAcquisition(num_frames, 0, True)
        print(f"Recording started on {camera_device}.")
        
        # Initialize timing variables
        frame_interval = 1.0 / fps  # Time between frames at desired FPS
        next_frame_time = time.time()  # When to capture next frame
        saved_frames = 0
        
        log_buffer = []  # Buffer log entries
        log_flush_interval = 40  # Flush every N frames (2x per second)

        while True:
            if os.path.exists(stop_file):
                print("Stop file detected. Terminating recording.")
                break

            current_time = time.time()
            
            # Check if it's time for the next frame
            if current_time >= next_frame_time:
                # Clear the buffer to get the most recent frame
                while mmc.getRemainingImageCount() > 1:
                    mmc.popNextImage()
                
                if mmc.getRemainingImageCount() > 0:
                    image = mmc.popNextImage()  # Retrieve the next image
                    frame = np.reshape(image, (frame_height, frame_width))  # Reshape to 2D array
                    
                    # Convert to 8-bit using FAST bit shift (not slow division)
                    if bit_shift > 0:
                        frame_8bit = (frame >> bit_shift).astype(np.uint8)
                    else:
                        frame_8bit = frame.astype(np.uint8)
                    
                    # Save frame first (priority)
                    video_writer.write(frame_8bit)
                    saved_frames += 1

                    # Publish for display thread — single reference assignment, no lock needed
                    latest_frame[0] = frame_8bit
                    
                    # Buffer log entries and flush periodically
                    log_buffer.append(f"{global_stopwatch.get_elapsed_time():.2f}\t{saved_frames}\n")
                    if saved_frames % log_flush_interval == 0:
                        log_file.writelines(log_buffer)
                        log_file.flush()
                        log_buffer.clear()
                    
                    # Calculate next frame time - add frame_interval to the original next_frame_time
                    # This prevents drift that could occur if we used current_time + frame_interval
                    next_frame_time += frame_interval
                    
                    # If we've fallen way behind (e.g., due to system lag), reset timing
                    if next_frame_time < current_time - frame_interval:
                        next_frame_time = current_time + frame_interval
                        print("Warning: Video timing reset due to system lag")
            
            # Small sleep to prevent busy-waiting, but short enough to not miss frame times
            time.sleep(0.0005)
    finally:
        # Flush any remaining log entries
        if log_buffer:
            log_file.writelines(log_buffer)
            log_file.flush()
        
        # Process any remaining frames in the buffer before stopping
        print("Processing remaining frames in buffer...")
        while mmc.getRemainingImageCount() > 0:
            image = mmc.popNextImage()  # Retrieve the next image
            frame = np.reshape(image, (frame_height, frame_width))  # Reshape to 2D array
            
            # Convert to 8-bit using FAST bit shift
            if bit_shift > 0:
                frame_8bit = (frame >> bit_shift).astype(np.uint8)
            else:
                frame_8bit = frame.astype(np.uint8)
            
            video_writer.write(frame_8bit)  # Write frame to video
            num_frames += 1
            log_file.write(f"{global_stopwatch.get_elapsed_time():.2f}\t{num_frames}\n")
            log_file.flush()

        stop_display.set()
        display_thread.join(timeout=1.0)
        mmc.stopSequenceAcquisition()
        video_writer.release()
        log_file.close()
        print(f"Recording stopped on {camera_device}. Video saved at {out_filename}.")
        print(f"Frame log saved at {log_path}")
        #print(f"Total frames recorded: {num_frames}")

if __name__ == "__main__":
    main()