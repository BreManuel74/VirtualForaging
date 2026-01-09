"""
KaufmanModule: Shared classes for Virtual Foraging Task

Contains:
- TrialLogging
- DataGenerator
- TextureSwapper
- RewardOrPuff
- RewardCalculator
- Stopwatch + global_stopwatch (shared timing source)
"""

from __future__ import annotations

import os
import time
import random
from typing import Any, Dict

import numpy as np
import pandas as pd

from direct.task import Task
from direct.fsm.FSM import FSM
from global_stopwatch import Stopwatch

# Create a global stopwatch instance shared across the app
global_stopwatch = Stopwatch()

class TrialLogging:
    """
    Centralizes logging of trial-level events into a DataFrame and CSV.
    Provides append-style APIs for each event type.
    """
    def __init__(self, output_dir: str, max_trials: int = 1000) -> None:
        ts = int(time.time())
        self.csv_path = os.path.join(output_dir, f"{ts}trial_log.csv")
        # Initialize dataframe with known columns
        self.df = pd.DataFrame({
            'rounded_base_hallway_data': np.full(max_trials, np.nan),
            'rounded_stay_data': np.full(max_trials, np.nan),
            'rounded_go_data': np.full(max_trials, np.nan),
            'segments_to_wait': np.full(max_trials, np.nan),
            'texture_history': np.full(max_trials, np.nan, dtype=object),
            'go_texture_change_time': np.full(max_trials, np.nan),
            'stay_texture_change_time': np.full(max_trials, np.nan),
            'stay_zone_active_state': pd.array([pd.NA] * max_trials, dtype='boolean'),
            'segments_until_revert': np.full(max_trials, np.nan),
            'go_texture_revert_time': np.full(max_trials, np.nan),
            'stay_texture_revert_time': np.full(max_trials, np.nan),
            'probe_texture_history': np.full(max_trials, np.nan, dtype=object),
            'probe_time': np.full(max_trials, np.nan),
            'puff_event': np.full(max_trials, np.nan, dtype=object),
            'reward_event': np.full(max_trials, np.nan, dtype=object),
        })
        self.save()

    def save(self) -> None:
        self.df.to_csv(self.csv_path, index=False)

    def _append_value(self, column: str, value: Any) -> None:
        col = self.df[column]
        # Find first NaN slot
        idxs = np.where(pd.isna(col))[0]
        if len(idxs) == 0:
            return  # No space left; optionally could expand
        self.df.at[int(idxs[0]), column] = value
        self.save()

    def set_initial_distributions(self, base: np.ndarray, stay: np.ndarray, go: np.ndarray) -> None:
        # Fill from start with provided arrays
        self.df.loc[0:len(base)-1, 'rounded_base_hallway_data'] = base
        self.df.loc[0:len(stay)-1, 'rounded_stay_data'] = stay
        self.df.loc[0:len(go)-1, 'rounded_go_data'] = go
        self.save()

    # Event-specific helpers
    def log_texture_history(self, texture: str) -> None:
        self._append_value('texture_history', str(texture))

    def log_go_texture_change_time(self, t: float) -> None:
        self._append_value('go_texture_change_time', float(t))

    def log_stay_texture_change_time(self, t: float) -> None:
        self._append_value('stay_texture_change_time', float(t))

    def log_segments_until_revert(self, n: int) -> None:
        self._append_value('segments_until_revert', int(n))

    def log_segments_to_wait(self, n: int) -> None:
        self._append_value('segments_to_wait', int(n))

    def log_probe_texture(self, texture: str) -> None:
        self._append_value('probe_texture_history', str(texture))

    def log_probe_time(self, t: float) -> None:
        self._append_value('probe_time', float(t))

    def log_go_texture_revert_time(self, t: float) -> None:
        self._append_value('go_texture_revert_time', float(t))

    def log_stay_texture_revert_time(self, t: float) -> None:
        self._append_value('stay_texture_revert_time', float(t))

    def log_puff_event(self, t: float) -> None:
        self._append_value('puff_event', float(t))

    def log_reward_event(self, t: float) -> None:
        self._append_value('reward_event', float(t))

    def log_stay_zone_active_state(self, state: bool) -> None:
        self._append_value('stay_zone_active_state', state)

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

    def generate_gaussian_data(self, key: str, size: int = 250, min_value: float | None = None) -> np.ndarray:
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

class TextureSwapper:
    """
    Class to manage and apply textures to the walls. Operates on a Corridor instance.
    """
    def __init__(self, corridor):
        self.corridor = corridor

    def change_wall_textures(self, task=None):
        """
        Change the textures of the left and right walls to a randomly selected texture.
        """
        c = self.corridor
        # Cache frequently accessed textures
        stop_texture = c.stop_texture
        go_texture = c.go_texture
        
        # Choose stop or go based on configured probability
        selected_texture = stop_texture if random.random() < c.stop_texture_probability else go_texture
        
        # Log selected texture
        c.texture_history = np.append(c.texture_history, str(selected_texture))
        c.trial_logger.log_texture_history(str(selected_texture))

        if selected_texture == stop_texture:
            c.reward_zone_active = True if random.random() < c.stay_zone_reward_probability else False
            #print(f"Reward Zone Active: {c.reward_zone_active}")
            # Log if reward probability was true or false
            c.trial_logger.log_stay_zone_active_state(c.reward_zone_active)

        # Choose distribution based on selected texture
        stay_or_go_data = c.rounded_go_data if selected_texture == go_texture else c.rounded_stay_data

        # Determine length of special zone and side effects
        segments_until_revert = int(random.choice(stay_or_go_data))
        c.segments_until_revert = segments_until_revert
        c.zone_gap = (19 - segments_until_revert) if selected_texture == stop_texture else 0
        c.base.zone_length = segments_until_revert
        
        # Log length
        c.segments_until_revert_history = np.append(c.segments_until_revert_history, segments_until_revert)
        c.trial_logger.log_segments_until_revert(segments_until_revert)

        # Apply textures to appropriate segments
        if selected_texture == stop_texture:
            forward_left, forward_right = c.get_forward_segments_far_reward(c.segments_until_revert)
            num_segments = min(len(forward_left), len(forward_right))
            for i in range(num_segments):
                c.apply_texture(forward_left[i], selected_texture)
                c.apply_texture(forward_right[i], selected_texture)
                c.set_segment_flag(forward_right[i], True)
            if c.cave is True:
            # Add cave texture to end of stay zone
                cave_segments = 4
                cave_texture = c.cave_texture
                cave_left, cave_right = c.get_forward_segments_far_reward_cave(cave_segments)
                for i in range(min(len(cave_left), len(cave_right))):
                    c.apply_texture(cave_left[i], cave_texture)
                    c.apply_texture(cave_right[i], cave_texture)
                cave_floor, cave_ceiling = c.get_forward_segments_far_reward_floor_and_ceiling(cave_segments)
                for i in range(min(len(cave_floor), len(cave_ceiling))):
                    c.apply_texture(cave_floor[i], cave_texture)
                    c.apply_texture(cave_ceiling[i], cave_texture)
        else:
            middle_left, middle_right = c.get_forward_segments_near(c.segments_until_revert)
            num_segments = min(len(middle_left), len(middle_right))
            for i in range(num_segments):
                c.apply_texture(middle_left[i], selected_texture)
                c.apply_texture(middle_right[i], selected_texture)
                c.set_segment_flag(middle_right[i], True)
            if c.cave is True:
                # Add cave texture to end of go zone
                cave_segments =  4
                cave_texture = c.cave_texture
                cave_left, cave_right = c.get_forward_segments_far(cave_segments)
                #print(len(cave_left), len(cave_right))
                for i in range(min(len(cave_left), len(cave_right))):
                    c.apply_texture(cave_left[i], cave_texture)
                    c.apply_texture(cave_right[i], cave_texture)
                cave_floor, cave_ceiling = c.get_forward_segments_far_floor_and_ceiling(cave_segments)
                for i in range(min(len(cave_floor), len(cave_ceiling))):
                    c.apply_texture(cave_floor[i], cave_texture)
                    c.apply_texture(cave_ceiling[i], cave_texture)

        return Task.done
    
    def apply_probe_locked_texture(self, task=None):
        """Temporarily change both walls to a single/non-random probe texture, then revert later"""
        c = self.corridor
        selected_temporary_texture = c.locked_probe
        
        # Apply temporary texture to forward segments
        probe_left, probe_right = c.get_forward_segments_far_reward_cave(24)
        probe_segments = min(len(probe_left), len(probe_right))
        apply_texture = c.apply_texture
        set_probe_flag = c.set_probe_flag
        for i in range(probe_segments):
            apply_texture(probe_left[i], selected_temporary_texture)
            set_probe_flag(probe_left[i], True)
            apply_texture(probe_right[i], selected_temporary_texture)
            set_probe_flag(probe_right[i], True)

        # Log probe texture and time
        c.probe_texture_history = np.append(c.probe_texture_history, str(selected_temporary_texture))
        c.trial_logger.log_probe_texture(str(selected_temporary_texture))

        elapsed_time = global_stopwatch.get_elapsed_time()
        elapsed_rounded = round(elapsed_time, 2)
        c.probe_time_history = np.append(c.probe_time_history, elapsed_rounded)
        c.trial_logger.log_probe_time(elapsed_rounded)

        # Schedule revert
        c.base.doMethodLaterStopwatch(c.probe_duration, self.revert_probe_texture, "RevertProbeTexture")
        return Task.done

    def apply_probe_texture(self, task=None):
        """Temporarily change both walls to a neutral/probe texture, then revert later."""
        c = self.corridor
        temporary_wall_textures = [
            c.neutral_stim_1,
            c.neutral_stim_2,
            c.neutral_stim_3,
            c.neutral_stim_4,
        ]
        selected_temporary_texture = random.choice(temporary_wall_textures)

        # Apply temporary texture to forward segments
        probe_left, probe_right = c.get_forward_segments_far(18)
        probe_segments = min(len(probe_left), len(probe_right))
        apply_texture = c.apply_texture
        set_probe_flag = c.set_probe_flag
        for i in range(probe_segments):
            apply_texture(probe_left[i], selected_temporary_texture)
            set_probe_flag(probe_left[i], True)
            apply_texture(probe_right[i], selected_temporary_texture)
            set_probe_flag(probe_right[i], True)


        #print(f"Applied probe texture: {selected_temporary_texture}")

        # Log probe texture and time
        c.probe_texture_history = np.append(c.probe_texture_history, str(selected_temporary_texture))
        c.trial_logger.log_probe_texture(str(selected_temporary_texture))

        elapsed_time = global_stopwatch.get_elapsed_time()
        elapsed_rounded = round(elapsed_time, 2)
        c.probe_time_history = np.append(c.probe_time_history, elapsed_rounded)
        c.trial_logger.log_probe_time(elapsed_rounded)

        # Schedule revert
        c.base.doMethodLaterStopwatch(c.probe_duration, self.revert_probe_texture, "RevertProbeTexture")
        return Task.done

    def revert_probe_texture(self, task=None):
        """Revert temporary probe textures back to corridor's right_wall_texture."""
        c = self.corridor
        probe_left, probe_right = c.get_forward_segments_far_probe_revert(44)
        probe_segments = min(len(probe_left), len(probe_right))
        right_wall_texture = c.right_wall_texture
        apply_texture = c.apply_texture
        get_probe_flag = c.get_probe_flag
        set_probe_flag = c.set_probe_flag
        for i in range(probe_segments):
            if get_probe_flag(probe_right[i]) and get_probe_flag(probe_left[i]):
                apply_texture(probe_left[i], right_wall_texture)
                apply_texture(probe_right[i], right_wall_texture)
                set_probe_flag(probe_left[i], False)
                set_probe_flag(probe_right[i], False)
        floor, ceiling = c.get_forward_segments_far_floor_and_ceiling(48)
        number_of_segments = min(len(floor), len(ceiling))
        floor_texture = c.floor_texture
        ceiling_texture = c.ceiling_texture
        for i in range(number_of_segments):
            apply_texture(floor[i], floor_texture)
            apply_texture(ceiling[i], ceiling_texture)
        return Task.done

    def exit_special_zones(self, task=None):
        """Log zone exit and optionally schedule a probe texture swap."""
        c = self.corridor
        elapsed_time = global_stopwatch.get_elapsed_time()
        elapsed_rounded = round(elapsed_time, 2)
        c.texture_revert_history = np.append(c.texture_revert_history, elapsed_rounded)
        
        # Cache textures for comparison
        go_texture = c.go_texture
        stop_texture = c.stop_texture
        trial_logger = c.trial_logger
        
        # Determine last special texture type just exited by inspecting previous frame info from base
        try:
            last_tex = getattr(c.base, 'prev_current_texture', None)
            if last_tex == go_texture:
                trial_logger.log_go_texture_revert_time(elapsed_rounded)
            elif last_tex == stop_texture:
                trial_logger.log_stay_texture_revert_time(elapsed_rounded)
            else:
                # Fallback: if base doesn't have prev_current_texture, try current flag context
                middle_left, middle_right = c.get_middle_segments(4)
                if middle_right and len(middle_right) >= 3:
                    # If we are now neutral, we assume we exited whatever was active in base flags
                    # Use active zone flags from base
                    if getattr(c.base, 'active_puff_zone', False):
                        trial_logger.log_go_texture_revert_time(elapsed_rounded)
                    elif getattr(c.base, 'active_stay_zone', False):
                        trial_logger.log_stay_texture_revert_time(elapsed_rounded)
                    else:
                        # Unknown; default to GO to avoid missing data
                        trial_logger.log_go_texture_revert_time(elapsed_rounded)
                else:
                    trial_logger.log_go_texture_revert_time(elapsed_rounded)
        except Exception:
            trial_logger.log_go_texture_revert_time(elapsed_rounded)

        # Cache probe flags and base for conditional checks
        if c.probe and random.random() < c.probe_probability:
            base = c.base
            probe_onset = c.probe_onset
            if c.probe_lock is False:
                base.doMethodLaterStopwatch(probe_onset, self.apply_probe_texture, "ApplyProbeTexture")
            else:
                base.doMethodLaterStopwatch(probe_onset, self.apply_probe_locked_texture, "ApplyProbeTexture")
        return Task.done

    def schedule_texture_change(self) -> None:
        """Schedule the next texture change by computing segments to wait."""
        c = self.corridor
        if not hasattr(c, 'segments_until_revert'):
            c.segments_until_revert = 0

        segments_to_wait = int(random.choice(c.rounded_base_hallway_data))
        c.segments_to_wait_history = np.append(c.segments_to_wait_history, segments_to_wait)
        c.trial_logger.log_segments_to_wait(segments_to_wait)

        c.segments_until_texture_change = segments_to_wait + c.segments_until_revert + c.zone_gap

    def update_texture_change(self) -> None:
        """Check if a texture change is needed and update state accordingly."""
        c = self.corridor
        if getattr(c, 'segments_until_texture_change', 0) <= 0:
            # Trigger the texture change
            self.change_wall_textures(None)

            # Inspect new front segments
            middle_left, middle_right = c.get_middle_segments(4)
            if middle_right and len(middle_right) >= 3:
                new_front_texture = middle_right[2].getTexture().getFilename()
                current_segment_flag = c.get_segment_flag(middle_right[2])
                c.current_segment_flag = current_segment_flag

                if new_front_texture == c.go_texture and current_segment_flag is True:
                    base = c.base
                    enter_go_time = global_stopwatch.get_elapsed_time()
                    base.enter_go_time = enter_go_time
                    # Log GO change time
                    enter_go_rounded = round(enter_go_time, 2)
                    c.texture_time_history = np.append(c.texture_time_history, enter_go_rounded)
                    c.trial_logger.log_go_texture_change_time(enter_go_rounded)
                    base.active_puff_zone = True
                    base.exit = True
                    set_segment_flag = c.set_segment_flag
                    for node in c.right_segments:
                        set_segment_flag(node, False)

            # Schedule the next texture change
            self.schedule_texture_change()

class RewardOrPuff(FSM):
    """
    FSM to manage the reward or puff state.
    """
    def __init__(self, base, config: Dict[str, Any]) -> None:
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
        # Use centralized trial logger for logging events
        self.trial_logger = self.base.trial_logger
        self.accept('puff-event', self.request, ['Puff'])
        self.accept('reward-event', self.request, ['Reward'])
        self.accept('neutral-event', self.request, ['Neutral'])
        self.current_texture = None

    def enterPuff(self):
        """
        Enter the Puff state.
        """
        self.puff_history = np.append(self.puff_history, round(global_stopwatch.get_elapsed_time(), 2))
        # Log puff event via trial logger
        self.trial_logger.log_puff_event(round(global_stopwatch.get_elapsed_time(), 2))

        # Combine 1 and puff_duration into a single integer
        signal = int(f"1{self.puff_duration}")
        self.base.serial_output.send_signal(signal)
        #print("puff!")
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
        # Log reward event via trial logger
        self.trial_logger.log_reward_event(round(global_stopwatch.get_elapsed_time(), 2))

        # Send reward message through TCP client if connected
        if hasattr(self.base, 'tcp_client') and self.base.tcp_client:
            self.base.tcp_client.send_data("REWARD:")

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
        Transition to the Neutral state or keep checking while Puff is active.
        """
        try:
            # If we just rewarded, always go neutral
            if self.state == 'Reward':
                self.request('Neutral')
                return Task.done

            # If we are in Puff, check the wall texture in front
            if self.state == 'Puff':
                middle_left, middle_right = self.base.corridor.get_middle_segments(4)
                # Be defensive about indexing
                if middle_right and len(middle_right) >= 3:
                    current_tex = str(middle_right[2].getTexture().getFilename())
                    # If still in GO zone, keep Puff by re-scheduling the check
                    if current_tex == self.base.corridor.go_texture:
                        signal = int(f"1{self.puff_duration}")
                        self.base.serial_output.send_signal(signal)
                        self.trial_logger.log_puff_event(round(global_stopwatch.get_elapsed_time(), 2))
                        #print("puff!")
                        self.base.doMethodLaterStopwatch(self.puff_to_neutral_time, self._transitionToNeutral, 'return-to-neutral')
                        return Task.done

                # Otherwise, leave Puff and go Neutral
                self.request('Neutral')
               #print("Leaving Puff state, going to Neutral")
                return Task.done

        except Exception as e:
            print(f"_transitionToNeutral error: {e}")
            # On error, fall back to Neutral
            self.request('Neutral')

        return Task.done

class RewardCalculator:
    """
    A new class that initializes with ShowBase and a configuration file.
    """
    def __init__(self, base, config: Dict[str, Any]) -> None:
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
        
    def calculate_x(self, y: float, slope: float, intercept: float) -> float | None:
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
