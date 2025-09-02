#!/usr/bin/env python3
"""
Infinite Corridor using Panda3D

This script creates an infinite corridor effect with user-controlled forward/backward movement.

Features:
- Configurable parameters loaded from JSON
- Infinite corridor effect
- User-controlled movement
- [real-time] Data logging (timestamp, distance, speed)

The corridor consists of left, right, ceiling, and floor segments.
It uses the Panda3D CardMaker API to generate flat geometry for the corridor's four faces.
An infinite corridor/hallway effect is simulated by recycling the front segments to the back when the player moves forward. 


Configuration parameters are loaded from a JSON file "conf.json".

Author: Jake Gronemeyer
Date: 2025-02-23
Version: 0.2
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
import numpy as np
from typing import Any, Dict
from dataclasses import dataclass
from datetime import datetime

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import CardMaker, NodePath, Texture, WindowProperties, Fog, GraphicsPipe
from direct.showbase import DirectObject
from direct.fsm.FSM import FSM
import pandas as pd

class Stopwatch:
    """
    A simple stopwatch class to measure elapsed time.
    """
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        """
        Start or resume the stopwatch.
        """
        if not self.running:
            self.start_time = time.time() - self.elapsed_time
            self.running = True

    def stop(self):
        """
        Stop the stopwatch and record the elapsed time.
        """
        if self.running:
            self.elapsed_time = time.time() - self.start_time
            self.running = False

    def reset(self):
        """
        Reset the stopwatch to zero.
        """
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def get_elapsed_time(self):
        """
        Get the elapsed time in seconds.
        
        Returns:
            float: The elapsed time in seconds.
        """
        if self.running:
            return time.time() - self.start_time
        return self.elapsed_time

# Create a global stopwatch instance
global_stopwatch = Stopwatch()

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

class DataGenerator:
    """
    A class to generate Gaussian data based on configuration parameters.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DataGenerator with configuration.

        Args:
            config (dict): Configuration dictionary containing Gaussian parameters.
        """
        self.config = config

    def generate_gaussian_data(self, key: str, size: int = 250, min_value: float = None) -> np.ndarray:
        """
        Generate Gaussian data based on the configuration.

        Args:
            key (str): The key in the configuration for the Gaussian parameters.
            size (int): The number of samples to generate.
            min_value (float): Minimum value to accept (optional).

        Returns:
            np.ndarray: Rounded Gaussian data.
        """
        loc = self.config[key]["loc"]
        scale = self.config[key]["scale"]

        if min_value is not None:
            data = []
            while len(data) < size:
                sample = np.random.normal(loc=loc, scale=scale)
                if sample >= min_value:
                    data.append(sample)
            return np.round(data)
        else:
            data = np.random.normal(loc=loc, scale=scale, size=size)
            return np.round(data)
        
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

        self.texture_history = self.base.texture_history
        self.texture_time_history = self.base.texture_time_history
        self.segments_until_revert_history = self.base.segments_until_revert_history
        self.segments_to_wait_history = self.base.segments_to_wait_history
        self.probe_texture_history = self.base.probe_texture_history
        self.probe_time_history = self.base.probe_time_history
        self.texture_revert_history = self.base.texture_revert_history
        self.trial_df = self.base.trial_df
        self.trial_df.to_csv = self.base.trial_df.to_csv 
        self.trial_csv_path = self.base.trial_csv_path

        self.segment_length: float = config["segment_length"]
        self.corridor_width: float = config["corridor_width"]
        self.wall_height: float = config["wall_height"]
        self.num_segments: int = config["num_segments"]
        self.left_wall_texture: str = config["left_wall_texture"]
        self.right_wall_texture: str = config["right_wall_texture"]
        self.ceiling_texture: str = config["ceiling_texture"]
        self.floor_texture: str = config["floor_texture"]
        self.go_texture: str = config["go_texture"]
        self.neutral_stim_1 = config["neutral_stim_1"]
        self.neutral_stim_2 = config["neutral_stim_2"]
        self.neutral_stim_3 = config["neutral_stim_3"]
        self.neutral_stim_4 = config["neutral_stim_4"]
        self.stop_texture = config["stop_texture"]
        self.probe_onset = config["probe_onset"]
        self.probe_duration = config["probe_duration"]
        self.probe_probability = config.get("probe_probability", 1.0)  # Default to 100% if not specified
        self.stop_texture_probability = config.get("stop_texture_probability", 0.5)  # Default to 50% if not specified
        
        # Create a parent node for all corridor segments.
        self.parent: NodePath = base.render.attachNewNode("corridor")
        
        # Separate lists for each face.
        self.left_segments: list[NodePath] = []
        self.right_segments: list[NodePath] = []
        self.ceiling_segments: list[NodePath] = []
        self.floor_segments: list[NodePath] = []
        
        self.build_segments()
        
        # Add a task to change textures at a random interval.
        self.schedule_texture_change()

        # Initialize attributes
        self.segments_until_revert = 0  # Ensure this attribute exists
        self.texture_change_scheduled = False  # Flag to track texture change scheduling

    def build_segments(self) -> None:
        """ 
        Build the initial corridor segments using CardMaker.
        """
        for i in range(-self.num_segments // 2, self.num_segments // 2):  # Adjust range to include negative indices
            segment_start: float = i * self.segment_length
            
            # ==== Left Wall:
            cm_left: CardMaker = CardMaker("left_wall")
            cm_left.setFrame(0, self.segment_length, 0, self.wall_height)
            left_node: NodePath = self.parent.attachNewNode(cm_left.generate())
            left_node.setPos(-self.corridor_width / 2, segment_start, 0)
            left_node.setHpr(90, 0, 0)
            self.apply_texture(left_node, self.left_wall_texture)
            self.left_segments.append(left_node)
            
            # ==== Right Wall:
            cm_right: CardMaker = CardMaker("right_wall")
            cm_right.setFrame(0, self.segment_length, 0, self.wall_height)
            right_node: NodePath = self.parent.attachNewNode(cm_right.generate())
            right_node.setPos(self.corridor_width / 2, segment_start, 0)
            right_node.setHpr(-90, 0, 0)
            self.apply_texture(right_node, self.right_wall_texture)
            self.right_segments.append(right_node)
            
            # ==== Ceiling (Top):
            cm_ceiling: CardMaker = CardMaker("ceiling")
            cm_ceiling.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
            ceiling_node: NodePath = self.parent.attachNewNode(cm_ceiling.generate())
            ceiling_node.setPos(0, segment_start, self.wall_height)
            ceiling_node.setHpr(0, 90, 0)
            self.apply_texture(ceiling_node, self.ceiling_texture)
            self.ceiling_segments.append(ceiling_node)
            
            # ==== Floor (Bottom):
            cm_floor: CardMaker = CardMaker("floor")
            cm_floor.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
            floor_node: NodePath = self.parent.attachNewNode(cm_floor.generate())
            floor_node.setPos(0, segment_start, 0)
            floor_node.setHpr(0, -90, 0)
            self.apply_texture(floor_node, self.floor_texture)
            self.floor_segments.append(floor_node)
            
    def apply_texture(self, node: NodePath, texture_path: str) -> None:
        """
        Load and apply the texture to a geometry node.
        
        Parameters:
            node (NodePath): The node to which the texture will be applied.
        """
        texture: Texture = self.base.loader.loadTexture(texture_path)
        node.setTexture(texture)
        
    def recycle_segment(self, direction: str) -> None:
        """
        Recycle the front segments by repositioning them to the end of the corridor.
        This is called when the player has advanced by one segment length.
        """
        if direction == "forward":
            # Calculate new base Y position from the last segment in the left wall.
            new_y: float = self.left_segments[-1].getY() + self.segment_length

            # Recycle left wall segment.
            left_seg: NodePath = self.left_segments.pop(0)
            left_seg.setY(new_y)
            self.left_segments.append(left_seg)

            # Recycle right wall segment.
            right_seg: NodePath = self.right_segments.pop(0)
            right_seg.setY(new_y)
            self.right_segments.append(right_seg)

            # Recycle ceiling segment.
            ceiling_seg: NodePath = self.ceiling_segments.pop(0)
            ceiling_seg.setY(new_y)
            self.ceiling_segments.append(ceiling_seg)

            # Recycle floor segment.
            floor_seg: NodePath = self.floor_segments.pop(0)
            floor_seg.setY(new_y)
            self.floor_segments.append(floor_seg)

        elif direction == "backward":
            # Calculate new base Y position from the first segment in the left wall.
            new_y: float = self.left_segments[0].getY() - self.segment_length

            # Recycle left wall segment.
            left_seg: NodePath = self.left_segments.pop(-1)
            left_seg.setY(new_y)
            self.left_segments.insert(0, left_seg)

            # Recycle right wall segment.
            right_seg: NodePath = self.right_segments.pop(-1)
            right_seg.setY(new_y)
            self.right_segments.insert(0, right_seg)

            # Recycle ceiling segment.
            ceiling_seg: NodePath = self.ceiling_segments.pop(-1)
            ceiling_seg.setY(new_y)
            self.ceiling_segments.insert(0, ceiling_seg)

            # Recycle floor segment.
            floor_seg: NodePath = self.floor_segments.pop(-1)
            floor_seg.setY(new_y)
            self.floor_segments.insert(0, floor_seg)
            
    def change_wall_textures(self, task: Task = None) -> Task:
        """
        Change the textures of the left and right walls to a randomly selected texture.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Define a list of possible wall textures with weighted probabilities
        # Use configurable probability for stop_texture, remainder for go_texture
        if random.random() < self.stop_texture_probability:
            selected_texture = self.stop_texture
        else:
            selected_texture = self.go_texture
        
        # Append to numpy array
        self.texture_history = np.append(self.texture_history, str(selected_texture))

        textures = np.full(len(self.trial_df), np.nan, dtype=object)
        textures[:len(self.texture_history)] = self.texture_history
        self.trial_df['texture_history'] = textures
        self.trial_df.to_csv(self.trial_csv_path, index=False)
        
        # Apply the selected texture to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, selected_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, selected_texture)
        
        # Print the elapsed time since the corridor was initialized
        elapsed_time = global_stopwatch.get_elapsed_time()
        self.texture_time_history = np.append(self.texture_time_history, round(elapsed_time, 2))
        times = np.full(len(self.trial_df), np.nan)
        times[:len(self.texture_time_history)] = self.texture_time_history
        self.trial_df['texture_change_time'] = times
        self.trial_df.to_csv(self.trial_csv_path, index=False)
        
        # Determine the stay_or_go_data based on the selected texture
        if selected_texture == self.go_texture:
            stay_or_go_data = self.rounded_go_data
        else:
            stay_or_go_data = self.rounded_stay_data
        
        # Set the counter for segments to revert textures using a random value from stay_or_go_data
        self.segments_until_revert = int(random.choice(stay_or_go_data))
        self.base.zone_length = self.segments_until_revert
        
        # Write the segments_until_revert value to the trial_data file
        self.segments_until_revert_history = np.append(self.segments_until_revert_history, int(self.segments_until_revert))
        length = np.full(len(self.trial_df), np.nan, dtype=float)
        length[:len(self.segments_until_revert_history)] = self.segments_until_revert_history
        self.trial_df['segments_until_revert'] = length
        self.trial_df.to_csv(self.trial_csv_path, index=False)
        
        # Return Task.done if task is None
        return Task.done if task is None else task.done

    def change_wall_textures_temporarily_once(self, task: Task = None) -> Task:
        """
        Temporarily change the wall textures for 1 second and then revert them back.
        This method ensures the temporary texture change happens only once.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Define a list of possible wall textures
        temporary_wall_textures = [
            self.neutral_stim_1,   # Texture 1
            self.neutral_stim_2,   # Texture 2
            self.neutral_stim_3,   # Texture 3
            self.neutral_stim_4,   # Texture 4
        ]

        # Randomly select a texture
        selected_temporary_texture = random.choice(temporary_wall_textures)

        self.probe_texture_history = np.append(self.probe_texture_history, str(selected_temporary_texture))

        probe_textures = np.full(len(self.trial_df), np.nan, dtype=object)
        probe_textures[:len(self.probe_texture_history)] = self.probe_texture_history
        self.trial_df['probe_texture_history'] = probe_textures
        self.trial_df.to_csv(self.trial_csv_path, index=False)

        ## Print the elapsed time since the corridor was initialized
        elapsed_time = global_stopwatch.get_elapsed_time() 
        self.probe_time_history = np.append(self.probe_time_history, round(elapsed_time, 2))
        
        probe_times = np.full(len(self.trial_df), np.nan)
        probe_times[:len(self.probe_time_history)] = self.probe_time_history
        self.trial_df['probe_time'] = probe_times
        self.trial_df.to_csv(self.trial_csv_path, index=False)

        # Apply the selected texture to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, selected_temporary_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, selected_temporary_texture)
        
        # Schedule a task to revert the textures back after 1 second
        self.base.doMethodLaterStopwatch(self.probe_duration, self.revert_temporary_textures, "RevertWallTextures")
        
        # Do not reset the texture_change_scheduled flag here to prevent repeated scheduling
        return Task.done if task is None else task.done
    
    def revert_temporary_textures(self, task: Task = None) -> Task:
        """
        Revert the temporary textures of the left and right walls back to their original textures.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Reapply the original textures to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, self.left_wall_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, self.right_wall_texture)
        
        # Return Task.done if task is None
        return Task.done if task is None else task.done

    def revert_wall_textures(self, task: Task = None) -> Task:
        """
        Revert the textures of the left and right walls to their original textures.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        elapsed_time = global_stopwatch.get_elapsed_time()
        self.texture_revert_history = np.append(self.texture_revert_history, round(elapsed_time, 2))

        revert_times = np.full(len(self.trial_df), np.nan)
        revert_times[:len(self.texture_revert_history)] = self.texture_revert_history
        self.trial_df['texture_revert'] = revert_times
        self.trial_df.to_csv(self.trial_csv_path, index=False)

        # Reapply the original textures to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, self.left_wall_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, self.right_wall_texture)
        
        #Conditional to make probe optional
        if self.base.cfg.get("probe", True):
            # Configurable chance of calling the probe function
            if random.random() < self.probe_probability:
                # Schedule a task to change the wall textures temporarily after reverting
                self.base.doMethodLaterStopwatch(self.probe_onset, self.change_wall_textures_temporarily_once, "ChangeWallTexturesTemporarilyOnce")
        
        # Return Task.done if task is None
        return Task.done if task is None else task.done

    def schedule_texture_change(self) -> None:
        """
        Schedule the next texture change after a random number of wall segments are recycled.
        """
        # Ensure segments_until_revert is initialized
        if not hasattr(self, 'segments_until_revert'):
            self.segments_until_revert = 0

        # Randomly determine the number of segments after which to change the texture
        segments_to_wait = random.choice(self.rounded_base_hallway_data)

        # Append to numpy array
        self.segments_to_wait_history = np.append(self.segments_to_wait_history, int(segments_to_wait))

        segs = np.full(len(self.trial_df), np.nan)
        segs[:len(self.segments_to_wait_history)] = self.segments_to_wait_history
        self.trial_df['segments_to_wait'] = segs
        self.trial_df.to_csv(self.trial_csv_path, index=False)
        
        self.segments_until_texture_change = segments_to_wait + self.segments_until_revert

    def update_texture_change(self) -> None:
        """
        Check if the required number of segments has been recycled and change the texture if needed.
        """
        if self.segments_until_texture_change <= 0:
            # Trigger the texture change
            self.change_wall_textures(None)
            
            # Check if the new texture is the go texture
            new_front_texture = self.left_segments[0].getTexture().getFilename()
            if new_front_texture == self.go_texture:
                # Update the enter_go_time in the MousePortal instance
                self.base.enter_go_time = global_stopwatch.get_elapsed_time()
                #print(f"enter_go_time updated to {self.base.enter_go_time:.2f} seconds")
            elif new_front_texture == self.stop_texture:
                # Update the enter_stay_time in the MousePortal instance
                self.base.enter_stay_time = global_stopwatch.get_elapsed_time()
                #print(f"enter_stay_time updated to {self.base.enter_stay_time:.2f} seconds")

            # Schedule the next texture change
            self.schedule_texture_change()

        # Check if textures need to be reverted
        if hasattr(self, 'segments_until_revert') and self.segments_until_revert > 0:
            self.segments_until_revert -= 1
            if self.segments_until_revert == 0:
                self.revert_wall_textures(None)  # Revert textures

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
        render.setFog(self.fog)

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

class RewardOrPuff(FSM):
    """
    FSM to manage the reward or puff state.
    """
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        """
        Initialize the FSM with the base and configuration.

        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
        """
        FSM.__init__(self, "RewardOrPuff")
        self.base = base
        self.config = config
        self.puff_duration = config["puff_duration"]
        self.puff_to_neutral_time = config["puff_to_neutral_time"]
        self.reward_duration = self.base.reward_duration
        self.puff_history = self.base.puff_history
        self.reward_history = self.base.reward_history
        self.trial_df = self.base.trial_df
        self.trial_df.to_csv = self.base.trial_df.to_csv 
        self.trial_csv_path = self.base.trial_csv_path
        self.accept('puff-event', self.request, ['Puff'])
        self.accept('reward-event', self.request, ['Reward'])
        self.accept('neutral-event', self.request, ['Neutral'])

    def enterPuff(self):
        """
        Enter the Puff state.
        """
        self.puff_history = np.append(self.puff_history, round(global_stopwatch.get_elapsed_time(), 2))
        puff_times = np.full(len(self.trial_df), np.nan)
        puff_times[:len(self.puff_history)] = self.puff_history
        self.trial_df['puff_event'] = puff_times
        self.trial_df.to_csv(self.trial_csv_path, index=False)

        # Combine 1 and puff_duration into a single integer
        signal = int(f"1{self.puff_duration}")
        self.base.serial_output.send_signal(signal)
        self.base.doMethodLaterStopwatch(self.puff_to_neutral_time, self._transitionToNeutral, 'return-to-neutral')

    def exitPuff(self):
        """
        Exit the Puff state.
        """
        #print("Exiting Puff state")
        
    def enterReward(self):
        """
        Enter the Reward state.
        """
        self.reward_history = np.append(self.reward_history, round(global_stopwatch.get_elapsed_time(), 2))
        reward_times = np.full(len(self.trial_df), np.nan)
        reward_times[:len(self.reward_history)] = self.reward_history
        self.trial_df['reward_event'] = reward_times
        self.trial_df.to_csv(self.trial_csv_path, index=False)

        signal = int(f"2{self.reward_duration}")
        #print(signal)
        self.base.serial_output.send_signal(signal)
        self.base.doMethodLaterStopwatch(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitReward(self):
        """
        Exit the Reward state."""
        #print("Exiting Reward state.")

    def enterNeutral(self):
        """
        Enter the Neutral state."""
        #print("Entering Neutral state: waiting...")

    def exitNeutral(self):
        """
        Exit the Neutral state."""
        #print("Exiting Neutral state.")

    def _transitionToNeutral(self, task):
        """
        Transition to the Neutral state.
        - For the Reward state: Only transition if the wall texture is the original wall texture.
        - For the Puff state: Transition directly without checking the wall texture.
        """
        # Get the current texture of the left wall
        current_texture = self.base.corridor.left_segments[0].getTexture().getFilename()

        if self.state == 'Reward':
            # Transition to Neutral only if the wall texture matches the original wall texture
            if current_texture == self.base.corridor.left_wall_texture:
                self.request('Neutral')
        elif self.state == 'Puff':
            # Transition to Neutral directly without checking the wall texture
            self.request('Neutral')

        return Task.done

class RewardCalculator:
    """
    A new class that initializes with ShowBase and a configuration file.
    """
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        """
        Initialize the class with ShowBase and load the configuration.

        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.base = base
        self.config = config
        self.reward_file = config["reward_file"]

    def read_csv_to_dataframe(self) -> pd.DataFrame:
        """
        Read the reward CSV file (specified by the reward_file attribute) and load its contents into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(self.reward_file)  # Use the reward_file attribute as the path
            return df
        except Exception as e:
            print(f"Error reading CSV file {self.reward_file}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on failure

    def extract_linear_data(self, batch_id: int) -> pd.DataFrame:
        """
        Extract the y = mx + b data from the DataFrame for a specific batch_id.

        Assumes the CSV file has columns: 'dt', 'w', 'dw', 'b0', 'b1', 'batch_id',
        where 'b0' is the slope (m) and 'b1' is the y-intercept (b).

        Parameters:
            batch_id (int): The batch ID to filter the data.

        Returns:
            pd.DataFrame: A DataFrame with renamed columns filtered by batch_id.
        """
        try:
            df = self.read_csv_to_dataframe()
            if {'dt', 'w', 'dw', 'b0', 'b1', 'batch_id'}.issubset(df.columns):
                # Filter the DataFrame by batch_id
                df = df[df['batch_id'] == batch_id]

                # Rename the columns
                df = df.rename(columns={
                    'dt': 'time',
                    'w': 'water_volumes',
                    'dw': 'delta_water_volumes',
                    'b0': 'slope',
                    'b1': 'intercept'
                })
                return df[['time', 'water_volumes', 'delta_water_volumes', 'slope', 'intercept']]
            else:
                print("CSV file does not contain the required columns: 'dt', 'w', 'dw', 'b0', 'b1', 'batch_id'.")
                return pd.DataFrame()  # Return an empty DataFrame if columns are missing
        except Exception as e:
            print(f"Error extracting linear data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on failure
        
    def calculate_x(self, y: float, slope: float, intercept: float) -> float:
        """
        Calculate the x value for a given y using the linear equation y = mx + b.

        Parameters:
            y (float): The y value (reward amount) to plug into the equation.
            slope (float): The slope (m) of the line.
            intercept (float): The y-intercept (b) of the line.

        Returns:
            float: The calculated x value.
        """
        try:
            if slope == 0:
                raise ValueError("Slope cannot be zero for a valid linear equation.")
            x = (y - intercept) / slope
            #print(x)
            return x
        except Exception as e:
            print(f"Error calculating x for y={y}, slope={slope}, intercept={intercept}: {e}")
            return None

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
        
        # Load configuration from JSON
        with open(config_file, 'r') as f:
            self.cfg: Dict[str, Any] = load_config(config_file)

        # Initialize the RewardCalculator
        self.reward_calculator = RewardCalculator(self, self.cfg)

       # Initialize the DataGenerator
        data_generator = DataGenerator(self.cfg)

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

        self.trial_csv_path = os.path.join(os.environ.get("OUTPUT_DIR"), f"{int(time.time())}trial_log.csv")

        max_trials = 1000    
        self.segments_to_wait_history = np.array([], dtype=int)
        self.texture_history = np.array([], dtype=str)
        self.texture_time_history = np.array([], dtype=float)
        self.texture_revert_history = np.array([], dtype=float)
        self.segments_until_revert_history = np.array([], dtype=int)
        self.probe_texture_history = np.array([], dtype=str)
        self.probe_time_history = np.array([], dtype=float)
        self.puff_history = np.array([], dtype=float)
        self.reward_history = np.array([], dtype=float)
        trial_df = pd.DataFrame({
            'rounded_base_hallway_data': np.full(max_trials, np.nan),
            'rounded_stay_data': np.full(max_trials, np.nan),
            'rounded_go_data': np.full(max_trials, np.nan),
            'segments_to_wait': np.full(max_trials, np.nan, dtype=float),
            'texture_history': np.full(max_trials, np.nan, dtype=object),
            'texture_change_time': np.full(max_trials, np.nan, dtype=float),
            'segments_until_revert': np.full(max_trials, np.nan, dtype=float),
            'texture_revert': np.full(max_trials, np.nan, dtype=float),
            'probe_texture_history': np.full(max_trials, np.nan, dtype=object),
            'probe_time': np.full(max_trials, np.nan, dtype=float),
            'puff_event': np.full(max_trials, np.nan, dtype=object),
            'reward_event': np.full(max_trials, np.nan, dtype=object),
            })
        trial_df.to_csv(self.trial_csv_path, index=False)
        self.trial_df = trial_df

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
        
        # Set up key mapping for keyboard input
        self.key_map: Dict[str, bool] = {"forward": False, "backward": False}
        self.accept("arrow_up", self.set_key, ["forward", True])
        self.accept("arrow_up-up", self.set_key, ["forward", False])
        self.accept("arrow_down", self.set_key, ["backward", True])
        self.accept("arrow_down-up", self.set_key, ["backward", False])
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
        self.distance_since_recycle: float = 0.0
        
        # Movement speed (units per second)
        self.movement_speed: float = 10.0
        
        # Initialize treadmill logger
        treadmill_log_path = os.path.join(os.environ.get("OUTPUT_DIR"), f"{int(time.time())}treadmill.csv")
        self.treadmill_logger = TreadmillLogger(treadmill_log_path)
        capacitive_log_path = os.path.join(os.environ.get("OUTPUT_DIR"), f"{int(time.time())}capacitive.csv")
        self.capacitive_logger = CapacitiveSensorLogger(capacitive_log_path)

        # Add the update task
        self.taskMgr.add(self.update, "updateTask")
        
        # Initialize fog effect
        self.fog_effect = FogEffect(
            self,
            density=self.cfg["fog_density"],
            fog_color=(0.5, 0.5, 0.5)
        )
        
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
        self.time_spent_at_zero_speed = self.cfg["time_spent_at_zero_speed"]

    def doMethodLaterStopwatch(base, delay, func, name):
        target_time = global_stopwatch.get_elapsed_time() + delay
        def wrapper(task):
            if global_stopwatch.get_elapsed_time() >= target_time:
                func(task)
                return Task.done
            return Task.cont
        base.taskMgr.add(wrapper, name)

    def set_key(self, key: str, value: bool) -> None:
        """
        Update the key state for the given key.
        
        Parameters:
            key (str): The key identifier.
            value (bool): True if pressed, False if released.
        """
        self.key_map[key] = value
        
    def update(self, task: Task) -> Task:
        """
        Update the camera's position based on user input and recycle corridor segments
        when the player moves forward beyond one segment.
        
        Parameters:
            task (Task): The Panda3D task instance.
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        dt: float = globalClock.getDt()
        move_distance: float = 0.0
        
        # Update camera velocity based on key input
        if self.key_map["forward"]:
            self.camera_velocity = self.speed_scaling
        elif self.key_map["backward"]:
            self.camera_velocity = -self.speed_scaling
        else:
            self.camera_velocity = 0.0
        
        self.camera_velocity = (int(self.treadmill.data.speed) / self.cfg["treadmill_speed_scaling"])

        # Update camera position (movement along the Y axis)
        self.camera_position += self.camera_velocity * dt
        move_distance = self.camera_velocity * dt
        self.camera.setPos(0, self.camera_position, self.camera_height)
        
        # Recycle corridor segments when the camera moves beyond one segment length
        if move_distance > 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle >= self.segment_length:
                # Recycle the segment in the forward direction
                self.corridor.recycle_segment(direction="forward")
                self.distance_since_recycle -= self.segment_length
                self.corridor.segments_until_texture_change -= 1
                self.corridor.update_texture_change()

                # Check if the new front segment has the stay or go textures
                new_front_texture = self.corridor.left_segments[0].getTexture().getFilename()
                if new_front_texture == self.corridor.go_texture:
                    self.segments_with_go_texture += 1
                    #print(f"New segment with go texture counted: {self.segments_with_go_texture}")
                elif new_front_texture == self.corridor.stop_texture:
                    self.segments_with_stay_texture += 1
                    #print(f"New segment with stay texture counted: {self.segments_with_stay_texture}")
        
        elif move_distance < 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle <= -self.segment_length:
                self.corridor.recycle_segment(direction="backward")
                self.distance_since_recycle += self.segment_length

        # Log movement data (timestamp, distance, speed)
        self.treadmill_logger.log(self.treadmill.data)

        # FSM state transition logic
        # Dynamically get the current texture of the left wall
        selected_texture = self.corridor.left_segments[0].getTexture().getFilename()
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

        #print(self.treadmill.data.speed)

        if selected_texture == self.corridor.stop_texture:
            #print(self.zone_length)
            # Check if speed has been 0 for exactly 1 second
            speed_zero_for_one_second = (self.speed_zero_start_time is not None and 
                                       current_time >= self.speed_zero_start_time + self.time_spent_at_zero_speed)

            if (self.segments_with_stay_texture <= self.zone_length and 
                self.fsm.state != 'Reward' and 
                speed_zero_for_one_second):
                #print("Requesting Reward state")
                self.fsm.request('Reward')
        elif selected_texture == self.corridor.go_texture:
            #print(self.zone_length)
            if self.segments_with_go_texture <= self.zone_length and self.fsm.state != 'Puff' and current_time >= self.enter_go_time + (self.puff_time * self.zone_length):
                #print("Requesting Puff state")
                self.fsm.request('Puff')
        else:
            self.segments_with_go_texture = 0 
            self.segments_with_stay_texture = 0
            if self.fsm.state != 'Neutral':
                #print("Requesting Neutral state")
                self.fsm.request('Neutral')
        
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