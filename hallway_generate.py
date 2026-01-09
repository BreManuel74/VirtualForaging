#!/usr/bin/env python3
"""
Infinite Corridor using Panda3D that constantly generates and pops new segments as the user moves forward or backward.
Currently supports a stop/don't stop mouse behavioral task loaded from KaufmanModule.
Original Author: Jake Gronemeyer
Modified and upgraded by: Brenna Manuel

"""

import json
import sys
import csv
import os
import time
import serial
import random
import subprocess
import sys
import socket
import threading
import signal
import atexit
import numpy as np
from typing import Any, Dict
from dataclasses import dataclass

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import CardMaker, NodePath, Texture, WindowProperties, Fog, ClockObject
from direct.showbase import DirectObject
import pandas as pd

# Import shared classes from KaufmanModule
from KaufmanModule import (
    TrialLogging,
    DataGenerator,
    TextureSwapper,
    RewardOrPuff,
    RewardCalculator,
    global_stopwatch,
)

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.
    
    Parameters:
        config_file (str): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)
        
@dataclass
class CapacitiveData:
    """
    Represents a single capacitive sensor reading.
    """
    capacitive_value: int
    timestamp: int  # Add this line

    def __repr__(self):
        return f"CapacitiveData(capacitive_value={self.capacitive_value}, timestamp={self.timestamp})"

class CapacitiveSensorLogger(DirectObject.DirectObject):
    """
    Logs capacitive sensor data to a CSV file.
    """
    def __init__(self, filename: str) -> None:
        """
        Initialize the capacitive sensor logger.

        Args:
            filename (str): Path to the CSV file.
        """
        self.filename = filename
        self.fieldnames = ['arduino_timestamp','elapsed_time', 'capacitive_value']
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

        # Listen for capacitive data events
        self.accept('readCapacitive', self.log)

    def log(self, data: CapacitiveData) -> None:
        """
        Log the capacitive sensor data to the CSV file.

        Args:
            data (CapacitiveData): The capacitive sensor data to log.
        """
        self.writer.writerow({
            'arduino_timestamp': data.timestamp,  # Timestamp from the Arduino
            'elapsed_time': round(global_stopwatch.get_elapsed_time(), 2),  # Elapsed time from the global stopwatch
            'capacitive_value': data.capacitive_value
        })
        self.file.flush()

    def close(self) -> None:
        """
        Close the CSV file.
        """
        self.file.close()

@dataclass
class TreadmillData:
    """ Represents a single encoder reading."""
    timestamp: int
    distance: float
    speed: float

    def __repr__(self):
        return (f"TreadmillData(timestamp={self.timestamp}, "
                f"distance={self.distance:.3f} mm, speed={self.speed:.3f} mm/s)")

class TreadmillLogger:
    """
    Logs movement data to a CSV file.
    """
    def __init__(self, filename):
        """
        Initialize the treadmill logger.
        
        Args:
            filename (str): Path to the CSV file.
        """
        self.filename = filename
        self.fieldnames = ['timestamp', 'distance', 'speed', 'global_time']
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def log(self, data: TreadmillData):
        self.writer.writerow({'timestamp': data.timestamp, 'distance': data.distance, 'speed': data.speed, 'global_time': round(global_stopwatch.get_elapsed_time(), 2)})
        self.file.flush()

    def close(self):
        self.file.close()

class Corridor:
    def __init__(self, base: ShowBase, config: Dict[str, Any],
                 rounded_base_hallway_data: np.ndarray,
                 rounded_stay_data: np.ndarray,
                 rounded_go_data: np.ndarray) -> None:
        """
        Initialize the corridor by creating segments for each face.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
            rounded_base_hallway_data (np.ndarray): Gaussian data for texture changes.
            rounded_stay_data (np.ndarray): Gaussian data for stay textures.
            rounded_go_data (np.ndarray): Gaussian data for go textures.
        """
        self.base = base
        self.config = config
        self.rounded_base_hallway_data = rounded_base_hallway_data
        self.rounded_stay_data = rounded_stay_data
        self.rounded_go_data = rounded_go_data

        # Initialize flags to track wall segments
        self.segment_flag_key = "segment_flag"
        self.probe_flag_key = "probe_flag"

        self.texture_history = self.base.texture_history
        self.texture_time_history = self.base.texture_time_history
        self.segments_until_revert_history = self.base.segments_until_revert_history
        self.segments_to_wait_history = self.base.segments_to_wait_history
        self.probe_texture_history = self.base.probe_texture_history
        self.probe_time_history = self.base.probe_time_history
        self.texture_revert_history = self.base.texture_revert_history
        # Trial logging references (centralized)
        self.trial_logger = self.base.trial_logger
        self.trial_df = self.trial_logger.df
        self.trial_csv_path = self.trial_logger.csv_path

        self.zone_gap = 0  # Initialize zone_gap
        
        self.segment_length: float = config["segment_length"]
        self.corridor_width: float = config["corridor_width"]
        self.wall_height: float = config["wall_height"]
        self.num_segments: int = config["num_segments"]
        self.left_wall_texture: str = config["left_wall_texture"]
        self.right_wall_texture: str = config["right_wall_texture"]
        self.ceiling_texture: str = config["ceiling_texture"]
        self.floor_texture: str = config["floor_texture"]
        self.go_texture: str = config["go_texture"]
        self.cave_texture: str = config.get("cave_texture", "assets/black.png")  # Default cave texture
        self.neutral_stim_1 = config["neutral_stim_1"]
        self.neutral_stim_2 = config["neutral_stim_2"]
        self.neutral_stim_3 = config["neutral_stim_3"]
        self.neutral_stim_4 = config["neutral_stim_4"]
        self.stop_texture = config["stop_texture"]
        self.probe_onset = config["probe_onset"]
        self.probe_duration = config["probe_duration"]
        self.probe_probability = config.get("probe_probability", 1.0)  # Default to 100% if not specified
        self.stop_texture_probability = config.get("stop_texture_probability", 0.5)  # Default to 50% if not specified
        self.probe = config.get("probe", False)  # Default to False if not specified

        # Create a parent node for all corridor segments.
        self.parent: NodePath = base.render.attachNewNode("corridor")
        self.view_distance: float = self.segment_length * self.num_segments
        
        # Separate lists for each face.
        self.left_segments: list[NodePath] = []
        self.right_segments: list[NodePath] = []
        self.ceiling_segments: list[NodePath] = []
        self.floor_segments: list[NodePath] = []
        
        self.build_initial_segments()
        
        # Initialize texture swapper and schedule first texture change
        self.texture_swapper = TextureSwapper(self)
        # Add a task to change textures at a random interval.
        self.texture_swapper.schedule_texture_change()

        # Initialize attributes
        self.segments_until_revert = 0  # Ensure this attribute exists
        self.texture_change_scheduled = False  # Flag to track texture change scheduling

        self.current_segment_flag = self.get_segment_flag(self.right_segments[0])

        self.reward_zone_active = False
        self.stay_zone_reward_probability = config.get("stay_zone_reward_probability", 1)  # Default to 100% if not specified

        self.probe_lock = config.get("probe_lock", False)  # Default to False if not specified
        self.locked_probe = None
        probe_textures = [
        self.neutral_stim_1,
            self.neutral_stim_2,
            self.neutral_stim_3,
            self.neutral_stim_4,
        ]
        if self.probe_lock == True:
            self.locked_probe = random.choice(probe_textures)

        self.cave = config.get("cave", False)  # Default to False if not specified

    def build_initial_segments(self) -> None:
        """ 
        Build the initial corridor segments centered around the camera.
        """
        start_pos = -(self.num_segments * self.segment_length) / 2
        for i in range(self.num_segments):
            position = start_pos + (i * self.segment_length)
            self._create_segment(position)
    
    def _create_segment(self, position: float) -> None:
        """
        Create a new corridor segment at the specified position.
        
        Parameters:
            position (float): The Y position where to create the segment
        """
        # Create left wall
        cm_left = CardMaker("left_wall")
        cm_left.setFrame(0, self.segment_length, 0, self.wall_height)
        left_node = self.parent.attachNewNode(cm_left.generate())
        left_node.setPos(-self.corridor_width / 2, position, 0)
        left_node.setHpr(90, 0, 0)
        self.apply_texture(left_node, self.left_wall_texture)
        left_node.setPythonTag(self.segment_flag_key, False)
        left_node.setPythonTag(self.probe_flag_key, False)
        self.left_segments.append(left_node)
        
        # Create right wall
        cm_right = CardMaker("right_wall")
        cm_right.setFrame(0, self.segment_length, 0, self.wall_height)
        right_node = self.parent.attachNewNode(cm_right.generate())
        right_node.setPos(self.corridor_width / 2, position, 0)
        right_node.setHpr(-90, 0, 0)
        self.apply_texture(right_node, self.right_wall_texture)
        right_node.setPythonTag(self.segment_flag_key, False)
        right_node.setPythonTag(self.probe_flag_key, False)
        self.right_segments.append(right_node)
        
        # Create ceiling
        cm_ceiling = CardMaker("ceiling")
        cm_ceiling.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
        ceiling_node = self.parent.attachNewNode(cm_ceiling.generate())
        ceiling_node.setPos(0, position, self.wall_height)
        ceiling_node.setHpr(0, 90, 0)
        self.apply_texture(ceiling_node, self.ceiling_texture)
        self.ceiling_segments.append(ceiling_node)
        
        # Create floor
        cm_floor = CardMaker("floor")
        cm_floor.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
        floor_node = self.parent.attachNewNode(cm_floor.generate())
        floor_node.setPos(0, position, 0)
        floor_node.setHpr(0, -90, 0)
        self.apply_texture(floor_node, self.floor_texture)
        self.floor_segments.append(floor_node)

    def _delete_segment(self, index: int) -> None:
        """
        Delete a corridor segment at the specified index.
        
        Parameters:
            index (int): Index of the segment to delete
        """
        self.left_segments[index].removeNode()
        self.right_segments[index].removeNode()
        self.ceiling_segments[index].removeNode()
        self.floor_segments[index].removeNode()
        
        # Remove from lists
        del self.left_segments[index]
        del self.right_segments[index]
        del self.ceiling_segments[index]
        del self.floor_segments[index]

    def set_segment_flag(self, node: NodePath, value: bool) -> None:
        node.setPythonTag(self.segment_flag_key, bool(value))

    def set_probe_flag(self, node: NodePath, value: bool) -> None:
        node.setPythonTag(self.probe_flag_key, bool(value))

    def get_probe_flag(self, node: NodePath) -> bool:
        return bool(node.getPythonTag(self.probe_flag_key) or False)

    def get_segment_flag(self, node: NodePath) -> bool:
        return bool(node.getPythonTag(self.segment_flag_key) or False)

    def update_corridor(self, camera_pos: float) -> None:
        """
        Update corridor segments based on camera position.
        Creates new segments ahead and removes segments that are too far behind.
        
        Parameters:
            camera_pos (float): Current camera Y position
        """
        # Calculate the range where segments should exist
        min_y = camera_pos - (self.view_distance / 2)
        max_y = camera_pos + (self.view_distance / 2)
        
        # Create list for storing new segments' positions
        positions_to_create = []
        
        # Add new segments behind if needed
        if not self.left_segments or self.left_segments[0].getY() > min_y:
            prev_pos = (self.left_segments[0].getY() - self.segment_length 
                    if self.left_segments 
                    else camera_pos)
            while prev_pos >= min_y:
                positions_to_create.append(prev_pos)
                prev_pos -= self.segment_length
                
        # Add new segments ahead if needed
        if not self.left_segments or self.left_segments[-1].getY() < max_y:
            next_pos = (self.left_segments[-1].getY() + self.segment_length 
                    if self.left_segments 
                    else camera_pos)
            while next_pos <= max_y:
                positions_to_create.append(next_pos)
                next_pos += self.segment_length
        
        # Create all new segments
        for pos in sorted(positions_to_create):
            self._create_segment(pos)
        
        # Remove segments that are too far behind
        while self.left_segments and self.left_segments[0].getY() < min_y:
            self._delete_segment(0)
            
        # Remove segments that are too far ahead
        while self.left_segments and self.left_segments[-1].getY() > max_y:
            self._delete_segment(-1)
        
        # Keep segments sorted by Y position
        if self.left_segments:
            segments = list(zip(self.left_segments, self.right_segments, 
                            self.ceiling_segments, self.floor_segments))
            segments.sort(key=lambda x: x[0].getY())
            
            self.left_segments = [s[0] for s in segments]
            self.right_segments = [s[1] for s in segments]
            self.ceiling_segments = [s[2] for s in segments]
            self.floor_segments = [s[3] for s in segments]
        
        # Ensure we don't exceed the maximum number of segments
        while len(self.left_segments) > self.num_segments:
            if abs(self.left_segments[0].getY() - camera_pos) > abs(self.left_segments[-1].getY() - camera_pos):
                self._delete_segment(0)
            else:
                self._delete_segment(-1)
            
    def apply_texture(self, node: NodePath, texture_path: str) -> None:
        """
        Load and apply the texture to a geometry node.
        
        Parameters:
            node (NodePath): The node to which the texture will be applied.
        """
        texture: Texture = self.base.loader.loadTexture(texture_path)
        node.setTexture(texture)
            
    def get_forward_segments_near(self, count: int) -> tuple[list[NodePath], list[NodePath]]:
        """
        Get segments starting from the one behind the camera and then
        moving forward for the specified count.
        Camera is between segments 12 and 13 in a 24-segment hallway.
        
        Parameters:
            count (int): Number of segments to return
            
        Returns:
            tuple[list[NodePath], list[NodePath]]: Selected left and right wall segments
        """
        # Sort both left and right segments by Y position (front to back)
        sorted_left = sorted(self.left_segments, key=lambda x: x.getY())
        sorted_right = sorted(self.right_segments, key=lambda x: x.getY())
        
        # Camera is between segments 12 and 13
        # Start from segment 12 (behind camera) and take 'count' segments forward
        start_index = 12
        end_index = min(start_index + count, len(sorted_left))
        
        # Get segments from behind camera forward
        selected_right = sorted_right[start_index:end_index]
        
        # Offset left segments forward by 1 (same as get_forward_segments_far)
        selected_left = sorted_left[start_index-1:end_index-1]

        return selected_left, selected_right

    def get_forward_segments_far(self, count: int) -> tuple[list[NodePath], list[NodePath]]:
        """Get the specified number of segments ahead of the camera."""
        # Sort both left and right segments
        sorted_left = sorted(self.left_segments, key=lambda x: x.getY())
        sorted_right = sorted(self.right_segments, key=lambda x: x.getY())
        
        # Take the furthest count right segments normally
        selected_right = sorted_right[-count:]
        
        # Take left segments one position closer
        selected_left = sorted_left[-(count+1):-1]  # Skip last segment
        
        # # Debug print
        # print("\nForward Segments Y positions:")
        # print("Left wall segments:", [round(seg.getY(), 2) for seg in selected_left])
        # print("Right wall segments:", [round(seg.getY(), 2) for seg in selected_right])
        
        return selected_left, selected_right
    
    def get_forward_segments_far_probe_revert(self, count: int) -> tuple[list[NodePath], list[NodePath]]:
        """Get the specified number of segments ahead of the camera for probe revert."""
        # Sort both left and right segments
        sorted_left = sorted(self.left_segments, key=lambda x: x.getY())
        sorted_right = sorted(self.right_segments, key=lambda x: x.getY())
        
        # Take the furthest count right segments normally
        selected_right = sorted_right[-count:]
        
        # Take left segments one position closer
        selected_left = sorted_left[-(count+1):]  # Skip last segment
        
        return selected_left, selected_right
    
    def get_forward_segments_far_floor_and_ceiling(self, count: int) -> tuple[list[NodePath], list[NodePath]]:
        """Get the specified number of floor and ceiling segments ahead of the camera."""
        # Sort both floor and ceiling segments
        sorted_floor = sorted(self.floor_segments, key=lambda x: x.getY())
        sorted_ceiling = sorted(self.ceiling_segments, key=lambda x: x.getY())
        
        # Take the furthest count floor and ceiling segments normally
        selected_floor = sorted_floor[-count:]
        selected_ceiling = sorted_ceiling[-count:]
        
        return selected_floor, selected_ceiling
    
    def get_middle_segments(self, count: int) -> tuple[list[NodePath], list[NodePath]]:
        """Get segments centered around the player, using left wall Y positions as reference."""
        # Sort both left and right segments
        sorted_left = sorted(self.left_segments, key=lambda x: x.getY())
        sorted_right = sorted(self.right_segments, key=lambda x: x.getY())

        # Calculate middle index
        middle_left_idx = len(sorted_left) // 2
        middle_right_idx = len(sorted_right) // 2

        # Calculate start and end indices to get segments centered around the middle
        start_left = max(0, middle_left_idx - count // 2)
        end_left = start_left + count
        start_right = max(0, middle_right_idx - count // 2)
        end_right = start_right + count

        selected_left = sorted_left[start_left:end_left]
        selected_right = sorted_right[start_right:end_right]

        return selected_left, selected_right  

class FogEffect:
    """
    Class to manage and apply fog to the scene.
    """
    def __init__(self, base: ShowBase, fog_color, density):
        """
        Initialize the fog effect.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            fog_color (tuple): RGB color for the fog (default is white).
            near_distance (float): The near distance where the fog starts.
            far_distance (float): The far distance where the fog completely obscures the scene.
        """
        self.base = base
        self.fog = Fog("fog")
        base.setBackgroundColor(fog_color)
        
        # Set fog color.
        self.fog.setColor(*fog_color)
        
        # Set the density for the fog.
        self.fog.setExpDensity(density)
        
        # Attach the fog to the root node to affect the entire scene.
        self.base.render.setFog(self.fog)

class SerialInputManager(DirectObject.DirectObject):
    """
    Manages serial input via the pyserial interface.
    
    This class abstracts the serial connection and starts a thread that listens
    for serial data from the Teensy Board and Arduino.
    """
    def __init__(self, teensy_port: str, teensy_baudrate: int = 57600, 
                 arduino_serial: serial.Serial = None, 
                 messenger: DirectObject = None, test_mode: bool = False, test_csv_path: str = None) -> None:
        self.teensy_port = teensy_port
        self.teensy_baudrate = teensy_baudrate
        self.arduino_serial = arduino_serial  # Use the shared instance
        self.test_mode = test_mode
        self.test_file = None
        self.test_reader = None
        self.test_data = None
        self.teensy_serial = None

        # Initialize Teensy connection
        if self.test_mode:
            try:
                self.test_file = open(test_csv_path, 'r')
                self.test_reader = csv.reader(self.test_file)
                next(self.test_reader)  # Skip header
            except Exception as e:
                print(f"Failed to open {test_csv_path}: {e}")
                raise
        else:
            try:
                self.teensy_serial = serial.Serial(self.teensy_port, self.teensy_baudrate, timeout=1)
            except serial.SerialException as e:
                print(f"{self.__class__}: Failed to open Teensy serial port {self.teensy_port}: {e}")
                raise

        self.accept('readSerial', self._store_data)
        self.accept('readCapacitive', self._store_capacitive_data)
        self.data = TreadmillData(0, 0.0, 0.0)
        self.capacitive_data = CapacitiveData(capacitive_value=0, timestamp=0)
        self.messenger = messenger

    def _store_data(self, data: TreadmillData):
        self.data = data

    def _store_capacitive_data(self, data: CapacitiveData):
        self.capacitive_data = data

    def _read_teensy_serial(self, task: Task) -> Task:
        """Internal loop for continuously reading lines from the Teensy or test CSV file."""
        if self.test_mode:
            # Read data from the test CSV file
            try:
                line = next(self.test_reader)
                #print("Test mode line:", line)  # <-- Add this
                if line:
                    data = self._parse_line(','.join(line))
                    #print("Parsed data:", data)  # <-- Add this
                    if data:
                        self.messenger.send("readSerial", [data])
                return Task.cont
            except StopIteration:
                # End of the CSV file; stop the task
                print("End of test.csv reached.")
                return Task.done
        elif self.teensy_serial and getattr(self.teensy_serial, 'is_open', False):
            try:
                raw_line = self.teensy_serial.readline()
                line = raw_line.decode('utf-8', errors='replace').strip()
                if line:
                    data = self._parse_line(line)
                    if data:
                        self.messenger.send("readSerial", [data])
                return Task.cont
            except Exception as e:
                print(f"Serial read error: {e}")
                return Task.done
        else:
            # Serial port is closed or None, stop the task
            return Task.done

    def _read_arduino_serial(self, task: Task) -> Task:
        """Internal loop for continuously reading lines from the Arduino."""
        if self.arduino_serial:
            raw_line = self.arduino_serial.readline()
            line = raw_line.decode('utf-8', errors='replace').strip()
            try:
                # Expecting "timestamp,value"
                parts = line.split(',')
                if len(parts) == 2:
                    timestamp = int(parts[0].strip())
                    capacitive_value = int(parts[1].strip())
                    capacitive_data = CapacitiveData(capacitive_value=capacitive_value, timestamp=timestamp)
                    self.messenger.send("readCapacitive", [capacitive_data])
                else:
                    # Fallback: just value, no timestamp
                    capacitive_value = int(line)
                    capacitive_data = CapacitiveData(capacitive_value=capacitive_value, timestamp=0)
                    self.messenger.send("readCapacitive", [capacitive_data])
            except ValueError:
                pass  # Ignore non-integer lines
        return Task.cont

    def _parse_line(self, line: str):
        """
        Parse a line of serial output from the Teensy or test CSV.

        Expected line formats:
          - "timestamp,distance,speed"
          - "distance,speed"
          - "timestamp,distance,speed,global_time" (test mode)

        Args:
            line (str): A single line from the serial port or CSV.

        Returns:
            TreadmillData: An instance with parsed values, or None if parsing fails.
        """
        parts = line.split(',')
        try:
            if len(parts) == 4:
                # Format: timestamp, distance, speed, global_time (test mode)
                timestamp = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
                # global_time = float(parts[3].strip())  # If you want to use it elsewhere
                return TreadmillData(distance=distance, speed=speed, timestamp=timestamp)
            elif len(parts) == 3:
                # Format: timestamp, distance, speed
                timestamp = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
                return TreadmillData(distance=distance, speed=speed, timestamp=timestamp)
            elif len(parts) == 2:
                # Format: distance, speed
                distance = float(parts[0].strip())
                speed = float(parts[1].strip())
                return TreadmillData(distance=distance, speed=speed)
            else:
                # Likely a header or message line (non-data)
                return None
        except ValueError:
            # Non-numeric data (e.g., header info)
            return None

    def _parse_capacitive_line(self, line: str):
        """
        Parse a line of capacitive sensor data from the Arduino.

        Expected line format:
        - "timestamp,value" or just "value"

        Args:
            line (str): A single line from the serial port.

        Returns:
            CapacitiveData: An instance with the parsed values, or None if parsing fails.
        """
        try:
            parts = line.split(',')
            if len(parts) == 2:
                timestamp = int(parts[0].strip())
                capacitive_value = int(parts[1].strip())
            else:
                capacitive_value = int(line.strip())
                timestamp = 0
            return CapacitiveData(capacitive_value=capacitive_value, timestamp=timestamp)
        except ValueError:
            # If the line is not a valid integer, return None
            print("no data")
            return None

    def close(self):
        if self.test_mode and self.test_file:
            self.test_file.close()
        if not self.test_mode:
            if self.teensy_serial:
                self.teensy_serial.close()
            if self.arduino_serial:
                self.arduino_serial.close()

class SerialOutputManager(DirectObject.DirectObject):
    """
    Manages serial output to an Arduino.
    
    This class abstracts the serial connection and provides methods to send output signals
    based on input from the FSM class.
    """
    def __init__(self, arduino_serial: serial.Serial) -> None:
        self.serial = arduino_serial  # Use the shared instance

    def send_signal(self, signal: Any) -> None:
        """
        Send a signal to the Arduino.
        
        Parameters:
            signal (Any): The signal to send, e.g., 'reward', 'puff', or an integer.
        """
        if self.serial.is_open:
            try:
                if isinstance(signal, int):
                    # Send larger integers
                    self.serial.write(str(signal).encode() + b"\n")
                elif isinstance(signal, str):
                    # Send strings as UTF-8 encoded bytes
                    self.serial.write(f"{signal}".encode('utf-8'))
                else:
                    raise ValueError("Unsupported signal type. Must be int or str.")
                #print(f"Sent signal: {signal}")
            except Exception as e:
                print(f"Failed to send signal: {e}")
        else:
            print("Arduino serial port is not open.")

    def close(self) -> None:
        """Close the serial connection."""
        if self.serial:
            self.serial.close()
            #print("Arduino serial port closed.")

class TCPStreamClient(DirectObject.DirectObject):
    """
    TCP client to receive data from run_me.py TCP server.
    Handles dynamic level changing and other commands.
    Thread-safe with proper cleanup mechanisms.
    """
    def __init__(self, base: ShowBase, host='localhost', port=None):
        """
        Initialize the TCP client.
        
        Args:
            base: The Panda3D ShowBase instance
            host: Server hostname (default: localhost)
            port: Server port (read from TCP_SERVER_PORT environment variable if not provided)
        """
        self.base = base
        self.host = host
        self.port = port or int(os.environ.get("TCP_SERVER_PORT", "0"))
        self.socket = None
        self.connected = False
        self.receive_buffer = ""
        self.connection_thread = None
        self._shutdown_lock = threading.Lock()
        self._running = True
        self.trial_df = self.base.trial_df
        self.trial_df.to_csv = self.base.trial_df.to_csv 
        self.trial_csv_path = self.base.trial_csv_path
        self._send_lock = threading.Lock()
        
        if self.port == 0:
            print("Warning: No TCP server port specified. TCP client disabled.")
            return
            
        # Start connection in a separate thread to avoid blocking
        self.connection_thread = threading.Thread(
            target=self._connect_to_server, 
            daemon=True, 
            name="TCPClient"
        )
        self.connection_thread.start()
        
        # Add task to check for incoming data
        self.base.taskMgr.add(self._check_for_data, "TCPClientDataCheck")
    
    def _connect_to_server(self):
        """Connect to the TCP server in a separate thread with proper error handling."""
        try:
            if not self._running:
                return
                
            print(f"Attempting to connect to TCP server at {self.host}:{self.port}...")
            
            # Create socket and connect
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout for connection
            self.socket.connect((self.host, self.port))
            
            # Set to non-blocking for continuous reading
            self.socket.setblocking(False)
            
            with self._shutdown_lock:
                if self._running:
                    self.connected = True
                    print(f"Connected to TCP server at {self.host}:{self.port}")
                else:
                    # Shutdown was called while connecting
                    self.socket.close()
                    
        except Exception as e:
            print(f"Failed to connect to TCP server: {e}")
            with self._shutdown_lock:
                self.connected = False
    
    def _check_for_data(self, task):
        """Task to continuously check for incoming data with proper error handling."""
        if not self._running or not self.connected or not self.socket:
            return Task.cont
            
        try:
            # Try to receive data (non-blocking)
            data = self.socket.recv(1024).decode('utf-8')
            if data:
                self.receive_buffer += data
                # Process complete lines
                while '\n' in self.receive_buffer:
                    line, self.receive_buffer = self.receive_buffer.split('\n', 1)
                    if line.strip():
                        self._process_command(line.strip())
            elif not data:  # Empty data means connection closed
                print("TCP server closed connection")
                self.connected = False
                        
        except socket.error as e:
            # No data available or connection error
            import errno
            if hasattr(errno, 'EAGAIN') and hasattr(errno, 'EWOULDBLOCK'):
                if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    print(f"Socket error: {e}")
                    self.connected = False
            else:
                # Fallback for systems without errno constants
                error_msg = str(e).lower()
                if 'would block' not in error_msg and 'try again' not in error_msg:
                    print(f"Socket error: {e}")
                    self.connected = False
        except Exception as e:
            print(f"Error reading TCP data: {e}")
            self.connected = False
            
        return Task.cont
    
    def _process_command(self, command):
        """Process incoming commands from the TCP server."""
        try:
            print(f"Received TCP command: {command}")
            
            if command.startswith("CHANGE_LEVEL:"):
                # Extract level filename
                level_file = command[13:]  # Remove "CHANGE_LEVEL:" prefix
                self._change_level(level_file)
            elif command == "reward":
                # Trigger reward event
                self.base.messenger.send('reward-event')
            elif command == "puff":
                # Trigger puff event
                self.base.messenger.send('puff-event')
            elif command == "neutral":
                # Trigger neutral event
                self.base.messenger.send('neutral-event')
            else:
                print(f"Unknown command: {command}")
                
        except Exception as e:
            print(f"Error processing command '{command}': {e}")
    
    def _change_level(self, level_file):
        """Dynamically change the game level."""
        try:
            print(f"Changing level to: {level_file}")
            
            # Construct full path to level file
            level_path = os.path.join("Levels", level_file)
            
            if not os.path.exists(level_path):
                print(f"Level file not found: {level_path}")
                return
            
            # Load new configuration
            with open(level_path, 'r') as f:
                new_config = json.load(f)
            
            # Update the base configuration
            old_config = self.base.cfg.copy()
            self.base.cfg.update(new_config)
            
            # Regenerate distributions if they exist in the new config
            self._update_distributions_from_config()
            
            # Reload corridor with new configuration if needed
            self._reload_corridor_config()
            
            print(f"Successfully changed level to: {level_file}")
            
        except Exception as e:
            print(f"Error changing level to '{level_file}': {e}")
    
    def _reload_corridor_config(self):
        """Reload corridor configuration with new settings."""
        try:
            # Update corridor configuration
            corridor = self.base.corridor
            
            # Update corridor texture settings
            if hasattr(corridor, 'go_texture'):
                corridor.go_texture = self.base.cfg.get("go_texture", corridor.go_texture)
            if hasattr(corridor, 'stop_texture'):
                corridor.stop_texture = self.base.cfg.get("stop_texture", corridor.stop_texture)
            if hasattr(corridor, 'probe_onset'):
                corridor.probe_onset = self.base.cfg.get("probe_onset", corridor.probe_onset)
            if hasattr(corridor, 'probe_duration'):
                corridor.probe_duration = self.base.cfg.get("probe_duration", corridor.probe_duration)
            if hasattr(corridor, 'probe_probability'):
                corridor.probe_probability = self.base.cfg.get("probe_probability", corridor.probe_probability)
            if hasattr(corridor, 'stop_texture_probability'):
                corridor.stop_texture_probability = self.base.cfg.get("stop_texture_probability", corridor.stop_texture_probability)
            if hasattr(corridor, 'probe'):
                corridor.probe = self.base.cfg.get("probe", corridor.probe)

            # Update wall and surface textures
            corridor.left_wall_texture = self.base.cfg.get("left_wall_texture", corridor.left_wall_texture)
            corridor.right_wall_texture = self.base.cfg.get("right_wall_texture", corridor.right_wall_texture)
            corridor.floor_texture = self.base.cfg.get("floor_texture", corridor.floor_texture)
            corridor.ceiling_texture = self.base.cfg.get("ceiling_texture", corridor.ceiling_texture)
            
            # Update probe/neutral stimuli textures
            corridor.neutral_stim_1 = self.base.cfg.get("neutral_stim_1", corridor.neutral_stim_1)
            corridor.neutral_stim_2 = self.base.cfg.get("neutral_stim_2", corridor.neutral_stim_2)
            corridor.neutral_stim_3 = self.base.cfg.get("neutral_stim_3", corridor.neutral_stim_3)
            corridor.neutral_stim_4 = self.base.cfg.get("neutral_stim_4", corridor.neutral_stim_4)

            # Update MousePortal-level configuration settings (only attributes that exist on MousePortal)
            if hasattr(self.base, 'reward_time'):
                self.base.reward_time = self.base.cfg.get("reward_time", self.base.reward_time)
            if hasattr(self.base, 'puff_time'):
                self.base.puff_time = self.base.cfg.get("puff_time", self.base.puff_time)
            if hasattr(self.base, 'fog_color'):
                self.base.fog_color = self.base.cfg.get("fog_color", self.base.fog_color)
            if hasattr(self.base, 'time_spent_at_zero_speed'):
                self.base.time_spent_at_zero_speed = self.base.cfg.get("time_spent_at_zero_speed", self.base.time_spent_at_zero_speed)
            if hasattr(self.base, 'puff_duration'):
                self.base.puff_duration = self.base.cfg.get("puff_duration", self.base.puff_duration)
            if hasattr(self.base, 'reward_duration'):
                self.base.reward_duration = self.base.cfg.get("reward_duration", self.base.reward_duration)

            # Update RewardOrPuff FSM if it exists
            if hasattr(self.base, 'fsm') and self.base.fsm:
                self.base.fsm.reward_duration = self.base.cfg.get("reward_duration", getattr(self.base.fsm, 'reward_duration', None))
                self.base.fsm.puff_duration = self.base.cfg.get("puff_duration", getattr(self.base.fsm, 'puff_duration', None))
                self.base.fsm.puff_to_neutral_time = self.base.cfg.get("puff_to_neutral_time", getattr(self.base.fsm, 'puff_to_neutral_time', None))

            print("Configuration reloaded successfully")
            
        except Exception as e:
            print(f"Error reloading corridor configuration: {e}")
    
    def _update_distributions_from_config(self):
        """
        Update hallway and zone distributions from the current config, overriding existing arrays.
        Preserves logged history in the DataFrame CSV file.
        """
        try:
            # Check if distribution parameters exist in config
            has_base_hallway = "base_hallway_distribution" in self.base.cfg
            has_stay_zone = "stay_zone_distribution" in self.base.cfg
            has_go_zone = "go_zone_distribution" in self.base.cfg
            
            if not (has_base_hallway or has_stay_zone or has_go_zone):
                return  # No distribution parameters to update
            
            # Initialize DataGenerator if not already present
            if not hasattr(self.base, 'data_generator'):
                self.base.data_generator = DataGenerator(self.base.cfg)
            else:
                # Update the config reference in existing data generator
                self.base.data_generator.config = self.base.cfg
            
            # Generate new distributions (override, don't append)
            if has_base_hallway:
                self.base.rounded_base_hallway_data = self.base.data_generator.generate_gaussian_data(
                    "base_hallway_distribution", 
                    min_value=self.base.cfg.get("base_hallway_min_value")
                )
                # Update corridor's reference
                self.base.corridor.rounded_base_hallway_data = self.base.rounded_base_hallway_data
            
            if has_stay_zone:
                self.base.rounded_stay_data = self.base.data_generator.generate_gaussian_data(
                    "stay_zone_distribution", 
                    min_value=self.base.cfg.get("stay_zone_min_value")
                )
                # Update corridor's reference
                self.base.corridor.rounded_stay_data = self.base.rounded_stay_data
                #print(f"Updated stay_zone distribution. Length: {len(self.base.rounded_stay_data)}")
            
            if has_go_zone:
                self.base.rounded_go_data = self.base.data_generator.generate_gaussian_data(
                    "go_zone_distribution", 
                    min_value=self.base.cfg.get("go_zone_min_value")
                )
                # Update corridor's reference
                self.base.corridor.rounded_go_data = self.base.rounded_go_data
                #print(f"Updated go_zone distribution. Length: {len(self.base.rounded_go_data)}")
            
            # Append the new distribution data to the existing data in the DataFrame
            current_base_len = len(self.trial_df[pd.notna(self.trial_df['rounded_base_hallway_data'])])
            current_stay_len = len(self.trial_df[pd.notna(self.trial_df['rounded_stay_data'])])
            current_go_len = len(self.trial_df[pd.notna(self.trial_df['rounded_go_data'])])
            
            # For base hallway data
            if has_base_hallway:
                end_idx = current_base_len + len(self.base.rounded_base_hallway_data)
                self.trial_df.loc[current_base_len:end_idx-1, 'rounded_base_hallway_data'] = self.base.rounded_base_hallway_data
            
            # For stay zone data
            if has_stay_zone:
                end_idx = current_stay_len + len(self.base.rounded_stay_data)
                self.trial_df.loc[current_stay_len:end_idx-1, 'rounded_stay_data'] = self.base.rounded_stay_data
            
            # For go zone data
            if has_go_zone:
                end_idx = current_go_len + len(self.base.rounded_go_data)
                self.trial_df.loc[current_go_len:end_idx-1, 'rounded_go_data'] = self.base.rounded_go_data
            
            # Save the updated dataframe with appended distributions
            self.trial_df.to_csv(self.trial_csv_path, index=False)
            
            print("Distribution update from config completed successfully")
            
        except Exception as e:
            print(f"Error updating distributions from config: {e}")
    
    def send_data(self, data: str) -> bool:
        """
        Send data to the server in a thread-safe way.
        
        Args:
            data: The string data to send
            
        Returns:
            bool: True if data was sent successfully, False otherwise
        """
        if not self.connected or not self.socket:
            return False
            
        with self._send_lock:
            try:
                message = f"{data}\n"
                self.socket.send(message.encode('utf-8'))
                return True
            except Exception as e:
                print(f"Error sending data: {e}")
                return False
    
    def close(self):
        """Close the TCP connection with proper thread cleanup."""
        with self._shutdown_lock:
            if not self._running:
                return  # Already closing
                
            print("Closing TCP client...")
            self._running = False
            self.connected = False
            
            # Close socket
            if self.socket:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                    self.socket.close()
                except:
                    pass
                self.socket = None
            
            # Remove the data checking task
            try:
                self.base.taskMgr.remove("TCPClientDataCheck")
            except:
                pass  # Task might not exist
            
            # Wait for connection thread to finish
            if self.connection_thread and self.connection_thread.is_alive():
                print("Waiting for TCP client thread to finish...")
                self.connection_thread.join(timeout=3)
                if self.connection_thread.is_alive():
                    print("Warning: TCP client thread did not stop cleanly")
                else:
                    print("TCP client thread stopped successfully")
            
            print("TCP client connection closed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass

class MousePortal(ShowBase):
    """
    Main application class for the infinite corridor simulation.
    """
    def __init__(self, config_file) -> None:
        """
        Initialize the application, load configuration, set up the camera, user input,
        corridor geometry, and add the update task.
        """
        ShowBase.__init__(self)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
        
        # Load configuration from JSON
        with open(config_file, 'r') as f:
            self.cfg: Dict[str, Any] = load_config(config_file)

        # Initialize the RewardCalculator
        self.reward_calculator = RewardCalculator(self, self.cfg)

       # Initialize the DataGenerator
        data_generator = DataGenerator(self.cfg)

        self.rounded_base_hallway_data = np.array([], dtype=int)
        self.rounded_stay_data = np.array([], dtype=int)
        self.rounded_go_data = np.array([], dtype=int)

        # Generate Gaussian data using min_value from the configuration
        self.rounded_base_hallway_data = data_generator.generate_gaussian_data(
            "base_hallway_distribution", 
            min_value=self.cfg["base_hallway_min_value"]
        )
        self.rounded_stay_data = data_generator.generate_gaussian_data(
            "stay_zone_distribution", 
            min_value=self.cfg["stay_zone_min_value"]
        )
        self.rounded_go_data = data_generator.generate_gaussian_data(
            "go_zone_distribution", 
            min_value=self.cfg["go_zone_min_value"]
        )

        # Initialize histories
        self.segments_to_wait_history = np.array([], dtype=int)
        self.texture_history = np.array([], dtype=str)
        self.texture_time_history = np.array([], dtype=float)
        self.texture_revert_history = np.array([], dtype=float)
        self.segments_until_revert_history = np.array([], dtype=int)
        self.probe_texture_history = np.array([], dtype=str)
        self.probe_time_history = np.array([], dtype=float)
        self.puff_history = np.array([], dtype=float)
        self.reward_history = np.array([], dtype=float)

        # Set up centralized TrialLogging
        output_dir = os.environ.get("OUTPUT_DIR")
        self.trial_logger = TrialLogging(output_dir)
        self.trial_logger.set_initial_distributions(
            self.rounded_base_hallway_data,
            self.rounded_stay_data,
            self.rounded_go_data,
        )
        # Back-compat attributes
        self.trial_df = self.trial_logger.df
        self.trial_csv_path = self.trial_logger.csv_path

        # Retrieve the reward_amount and batch_id from the configuration
        reward_amount = self.cfg.get("reward_amount", 0.0)
        batch_id = os.environ.get("BATCH_ID")
        batch_id = int(batch_id)

        # Extract linear data for the specified batch_id
        linear_data = self.reward_calculator.extract_linear_data(batch_id)

        # Ensure the linear data is not empty
        if not linear_data.empty:
            # Use the first row of slope and intercept for calculation
            slope = linear_data.iloc[0]['slope']
            intercept = linear_data.iloc[0]['intercept']

            # Calculate the x value for the reward amount
            self.reward_duration = self.reward_calculator.calculate_x(reward_amount, slope, intercept)
            self.reward_duration = round(self.reward_duration)

            # Log or use the calculated x value
            #print(f"Calculated x value for reward amount {reward_amount} (batch_id {batch_id}): {self.reward_duration}")
        else:
            print(f"Failed to extract linear data for batch_id {batch_id}. Reward calculation skipped.")

        # Start the stopwatch
        global_stopwatch.start()

        # Pass the stopwatch start time to the subprocess via environment variable
        os.environ["STOPWATCH_START_TIME"] = str(global_stopwatch.start_time)

        # Start the video recording subprocess
        recorder_script = self.cfg.get("recorder_script_path", "thorcam.py")
        self.recorder_proc = subprocess.Popen([sys.executable, recorder_script])

        # Get the display width and height for both monitors
        pipe = self.win.getPipe()
        display_width = pipe.getDisplayWidth()
        display_height = pipe.getDisplayHeight()

        # Set window properties to span across both monitors
        wp = WindowProperties()
        wp.setSize(display_width * 2, display_height)  # Double the width for two monitors
        wp.setOrigin(0, 0)  # Start at the leftmost edge
        wp.setFullscreen(False)  # Ensure it's not in fullscreen mode
        self.setFrameRateMeter(False)
        self.disableMouse()  # Disable default mouse-based camera control
        wp.setCursorHidden(True)
        wp.setUndecorated(True)
        self.win.requestProperties(wp)
        
        # Initialize camera parameters
        self.camera_position: float = 0.0
        self.camera_velocity: float = 0.0
        self.speed_scaling: float = self.cfg.get("speed_scaling", 5.0)
        self.camera_height: float = self.cfg.get("camera_height", 2.0)  
        self.camera.setPos(0, self.camera_position, self.camera_height)
        self.camera.setHpr(0, 0, 0)
        
        self.accept('escape', self.userExit)

        # Set up shared Arduino serial connection
        self.arduino_serial = serial.Serial(
            self.cfg["arduino_port"],
            self.cfg["arduino_baudrate"],
            timeout=1
        )

        # Set up treadmill input
        self.treadmill = SerialInputManager(
            os.environ.get("TEENSY_PORT"),
            teensy_baudrate=self.cfg["teensy_baudrate"],
            arduino_serial=self.arduino_serial,  # Pass the shared instance
            messenger=self.messenger,
            test_mode=self.cfg.get("test_mode", False),
            test_csv_path= r'Kaufman_Project/BM15/Session 28/beh/1754413096treadmill.csv'
        )

        # Set up serial output to Arduino
        self.serial_output = SerialOutputManager(
            arduino_serial=self.arduino_serial  # Pass the shared instance
        )

        # Create corridor geometry and pass Gaussian data
        self.corridor: Corridor = Corridor(
            base=self,
            config=self.cfg,
            rounded_base_hallway_data=self.rounded_base_hallway_data,
            rounded_stay_data=self.rounded_stay_data,
            rounded_go_data=self.rounded_go_data
        )
        self.segment_length: float = self.cfg["segment_length"]
        
        # Initialize the RewardOrPuff FSM
        self.fsm = RewardOrPuff(self, self.cfg)
        self.zone_length = 0

        # Variable to track movement since last recycling
        self.distance_since_last_segment: float = 0.0
        
        # Movement speed (units per second)
        self.movement_speed: float = 10.0
        
        # Initialize treadmill logger
        treadmill_log_path = os.path.join(os.environ.get("OUTPUT_DIR"), f"{int(time.time())}treadmill.csv")
        self.treadmill_logger = TreadmillLogger(treadmill_log_path)
        capacitive_log_path = os.path.join(os.environ.get("OUTPUT_DIR"), f"{int(time.time())}capacitive.csv")
        self.capacitive_logger = CapacitiveSensorLogger(capacitive_log_path)

        # Add the update task
        self.taskMgr.add(self.update, "updateTask")

        self.fog_color = tuple(self.cfg["fog_color"])
        # Initialize fog effect
        self.fog_effect = FogEffect(
            self,
            density=self.cfg["fog_density"],
            fog_color=self.fog_color)
        
        # Set up task chain for serial input
        self.taskMgr.setupTaskChain(
            "teensySerialInput",
            numThreads=1,
            tickClock=None,
            threadPriority=None,
            frameBudget=None,
            frameSync=True,
            timeslicePriority=None
        )
        self.taskMgr.setupTaskChain(
            "arduinoSerialInput",
            numThreads=1,
            tickClock=None,
            threadPriority=None,
            frameBudget=None,
            frameSync=True,
            timeslicePriority=None
        )
        self.taskMgr.add(self.treadmill._read_teensy_serial, name="readTeensySerial", taskChain="teensySerialInput")
        self.taskMgr.add(self.treadmill._read_arduino_serial, name="readArduinoSerial", taskChain="arduinoSerialInput")

        # Enable verbose messaging
        #self.messenger.toggleVerbose()

        # Add an attribute to track the number of segments passed for the FSM logic
        self.segments_with_go_texture = 0
        self.segments_with_stay_texture = 0

        # Add attributes to store time points
        self.enter_go_time = 0.0
        self.enter_stay_time = 0.0
        self.speed_zero_start_time = None

        # Initialize TCP client for dynamic level changing
        self.tcp_client = TCPStreamClient(self)

        # Puff and reward 0-speed times
        self.time_spent_at_zero_speed = self.cfg["time_spent_at_zero_speed"]
        self.puff_zero_speed_time = self.cfg["puff_zero_speed_time"]
        
        # Initialize variables for tracking current texture and flag
        self.current_texture = self.corridor.right_segments[0].getTexture().getFilename()
        self.current_segment_flag = self.corridor.get_segment_flag(self.corridor.right_segments[0])
        self.active_stay_zone = False
        self.active_puff_zone = False
        self.fsm.current_state = 'Neutral'  # Start in neutral state
        self.exit = True
        # Track last frame's current texture to detect re-entries reliably
        self.prev_current_texture = self.current_texture
        # Track whether we just re-entered a special zone (so exit shouldn't schedule probe again)
        self.reentry_pending = False

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        try:
            self._cleanup()
        except:
            pass
        sys.exit(0)

    def doMethodLaterStopwatch(base, delay, func, name):
        target_time = global_stopwatch.get_elapsed_time() + delay
        def wrapper(task):
            if global_stopwatch.get_elapsed_time() >= target_time:
                func(task)
                return Task.done
            return Task.cont
        base.taskMgr.add(wrapper, name)

    def update(self, task: Task) -> Task:
        """
        Update the camera's position based on user input and recycle corridor segments
        when the player moves forward beyond one segment.
        
        Parameters:
            task (Task): The Panda3D task instance.
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        dt: float = ClockObject.getGlobalClock().getDt()
        move_distance: float = 0.0
        
        self.camera_velocity = (int(self.treadmill.data.speed) / self.cfg["treadmill_speed_scaling"])

        # Update camera position (movement along the Y axis)
        self.camera_position += self.camera_velocity * dt
        move_distance = self.camera_velocity * dt
        self.camera.setPos(0, self.camera_position, self.camera_height)

        # Update corridor
        self.corridor.update_corridor(self.camera_position)

        # Get current segments at camera position using get_middle_segments
        middle_left, middle_right = self.corridor.get_middle_segments(4)
        if middle_right:  # Using right wall instead of left
            self.current_texture = middle_right[2].getTexture().getFilename()
            self.former_texture = middle_right[1].getTexture().getFilename()
            self.current_segment_flag = self.corridor.get_segment_flag(middle_right[2])

            if self.current_texture == self.corridor.stop_texture and self.current_segment_flag == True:
                self.enter_stay_time = global_stopwatch.get_elapsed_time()
                self.texture_time_history = np.append(self.texture_time_history, round(self.enter_stay_time, 2))
                self.trial_logger.log_stay_texture_change_time(round(self.enter_stay_time, 2))
                self.active_stay_zone = True
                self.exit = True
                #print(f"Entered STAY zone at time: {self.enter_stay_time}")
                for node in self.corridor.right_segments:
                    self.corridor.set_segment_flag(node, False)

            # If we re-enter a special zone (GO or STOP) from neutral, allow a new exit log later
            if ((self.prev_current_texture == self.corridor.right_wall_texture or self.prev_current_texture == self.corridor.cave_texture)
                and (self.current_texture == self.corridor.go_texture or self.current_texture == self.corridor.stop_texture)
                and self.exit == False):
                self.exit = True
                self.reentry_pending = True
                # Re-log this re-entry time to GO/STAY-specific change column
                elapsed_time = global_stopwatch.get_elapsed_time()
                self.texture_time_history = np.append(self.texture_time_history, round(elapsed_time, 2))
                if self.current_texture == self.corridor.go_texture:
                    self.trial_logger.log_go_texture_change_time(round(elapsed_time, 2))
                elif self.current_texture == self.corridor.stop_texture:
                    self.trial_logger.log_stay_texture_change_time(round(elapsed_time, 2))
                

            if ((self.prev_current_texture == self.corridor.go_texture or self.prev_current_texture == self.corridor.stop_texture)
                and (self.current_texture == self.corridor.right_wall_texture or self.current_texture == self.corridor.cave_texture) and self.exit == True):
                if self.reentry_pending:
                    print("self.reentry pending is true")
                    # Log revert time locally without triggering probe again
                    elapsed_time = global_stopwatch.get_elapsed_time()
                    self.texture_revert_history = np.append(self.texture_revert_history, round(elapsed_time, 2))
                    # Use prev_current_texture to determine which column to log
                    if self.prev_current_texture == self.corridor.go_texture:
                        self.trial_logger.log_go_texture_revert_time(round(elapsed_time, 2))
                    elif self.prev_current_texture == self.corridor.stop_texture:
                        self.trial_logger.log_stay_texture_revert_time(round(elapsed_time, 2))
                else:
                    # First exit for this zone: use centralized handler (may schedule probe)
                    self.corridor.texture_swapper.exit_special_zones()
                    #print("called exit_special_zones()")
                #print("Exited a zone")
                self.reentry_pending = False
                self.exit = False
            
            # Update previous texture at the end of evaluation
            self.prev_current_texture = self.current_texture
                
        # Keep track of segments passed in either direction
        if move_distance > 0:
            self.distance_since_last_segment += move_distance
            while self.distance_since_last_segment >= self.segment_length:
                # Count a segment passed in forward direction
                self.distance_since_last_segment -= self.segment_length
                self.corridor.segments_until_texture_change -= 1
                self.corridor.texture_swapper.update_texture_change()

                if self.current_texture == self.corridor.go_texture and self.active_puff_zone == True:
                    self.segments_with_go_texture += 1
                    #print(f"New segment with go texture counted: {self.segments_with_go_texture}")
                elif self.current_texture == self.corridor.stop_texture and self.active_stay_zone == True:
                    self.segments_with_stay_texture += 1
                    #print(f"STAY zone - Segments: {self.segments_with_stay_texture}")

        elif move_distance < 0:
            self.distance_since_last_segment += move_distance
            while self.distance_since_last_segment <= -self.segment_length:
                # Count a segment passed in backward direction
                self.distance_since_last_segment += self.segment_length
                self.corridor.segments_until_texture_change += 1
                self.corridor.texture_swapper.update_texture_change()

        # Log movement data (timestamp, distance, speed)
        self.treadmill_logger.log(self.treadmill.data)

        # FSM state transition logic
        # Dynamically get the current texture of the left wall
        self.reward_time = self.cfg["reward_time"]
        self.puff_time = self.cfg["puff_time"]

        # Get the elapsed time from the global stopwatch
        current_time = global_stopwatch.get_elapsed_time()

        # Track when treadmill speed becomes 0 and reset if speed is not 0
        if self.treadmill.data.speed == 0:
            if self.speed_zero_start_time is None:
                self.speed_zero_start_time = current_time
        else:
            self.speed_zero_start_time = None

        #print(self.current_texture)

        if self.current_texture == self.corridor.stop_texture and self.active_stay_zone == True:
            #print(self.zone_length)
            # Check if speed has been 0 for set time
            speed_zero_duration = (self.speed_zero_start_time is not None and 
                                       current_time >= self.speed_zero_start_time + self.time_spent_at_zero_speed)
            # Check if enough time has been spent in the zone
            meets_time_requirement = current_time >= self.enter_stay_time + (self.reward_time * self.zone_length)
            
            if (self.segments_with_stay_texture <= self.zone_length and 
                self.fsm.state != 'Reward' and
                self.corridor.reward_zone_active == True and 
                (speed_zero_duration or meets_time_requirement)):  # Either condition can trigger reward
                #print("Requesting Reward state")
                self.fsm.request('Reward')
                self.active_stay_zone = False  # Reset stay zone flag after requesting reward

        elif self.current_texture == self.corridor.go_texture and self.active_puff_zone == True:
            #print(self.zone_length)
            # Check if speed has been 0 for set time
            speed_zero_duration = (self.speed_zero_start_time is not None and 
                                       current_time >= self.speed_zero_start_time + self.puff_zero_speed_time)
            # Check if enough time has been spent in the zone
            meets_time_requirement = current_time >= self.enter_go_time + (self.puff_time * self.zone_length)

            if (self.segments_with_go_texture <= self.zone_length and 
                self.fsm.state != 'Puff' and 
                (speed_zero_duration or meets_time_requirement)):
                #print("Requesting Puff state")
                self.fsm.request('Puff')
                self.active_puff_zone = False  # Reset puff zone flag after requesting puff

        else:
            self.segments_with_go_texture = 0 
            self.segments_with_stay_texture = 0
        
        return Task.cont

    def userExit(self):
        """Override userExit to ensure subprocess is stopped and cleanup occurs."""
        # Signal the subprocess to stop by creating the flag file
        with open("stop_recording.flag", "w") as f:
            f.write("stop")
        try:
            # Wait for the subprocess to finish
            self.recorder_proc.wait(timeout=10)
        except Exception as e:
            print(f"Recorder subprocess did not exit cleanly: {e}")
        # Remove the flag file if it exists
        if os.path.exists("stop_recording.flag"):
            os.remove("stop_recording.flag")
        # Call the original cleanup without recursion
        self._cleanup()
        # Exit the Panda3D app
        super().userExit()

    def close(self):
        self._cleanup()

    def _cleanup(self):
        # Remove serial reading tasks before closing ports
        self.taskMgr.remove("readTeensySerial")
        self.taskMgr.remove("readArduinoSerial")
        # Close TCP client connection
        if hasattr(self, 'tcp_client') and self.tcp_client:
            self.tcp_client.close()
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
        if self.treadmill:
            self.treadmill.close()
        if self.serial_output:
            self.serial_output.close()

if __name__ == "__main__":
    config_path = os.environ.get("LEVEL_CONFIG_PATH")
    if config_path is None:
        config_path = "levels/blank1_1.json"
    app = MousePortal(config_path)
    app.run()