"""
Simulator for the Moab plate+ball balancing device using concept selectors
"""

# pyright: strict

import os
import sys
import json
import time
from dotenv import load_dotenv, set_key
from pyrr import matrix33, vector
from typing import Dict, Any, List
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
)
from sim import moab_model
from concept_orchestration import ExportedBrainPredictor, launch_predictor_server
from policies import random_policy


dir_path = os.path.dirname(os.path.realpath(__file__))

class TemplateSimulatorSession:
    def __init__(
        self,
        modeldir: str = "sim",
        env_name: str = "moab-py-v5"
    ):
        """Simulator Interface with the Bonsai Platform
        Parameters
        ----------
        modeldir: str, optional
            directory where you sim folder lives
        env_name : str, optional
            Name of simulator interface, by default "Cartpole"
        """
        self.modeldir = modeldir
        self.env_name = env_name
        print("Using simulator file from: ", os.path.join(dir_path, self.modeldir))
        self.simulator = moab_model.MoabModel()
        C1_url = 'http://localhost:1111'
        C2_url = 'http://localhost:2222'
        self.C1 = ExportedBrainPredictor(predictor_url=C1_url, control_period=1)
        self.C2 = ExportedBrainPredictor(predictor_url=C2_url, control_period=1)
        
    def get_state(self) -> Dict[str, float]:
        """Called to retreive the current state of the simulator.
        
        Returns
        -------
        Dict[str, float]
            Returns float of current values from the simulator
        """
        return self.simulator.state()

    def clamp(self, val: float, min_val: float, max_val: float):
        """Clamp the values to defined ranges.
        """
        return min(max_val, max(min_val, val))

    def _set_velocity_for_speed_and_direction(self, speed: float, direction: float):
        """Set the direction and speed.       
        """
        # Get the heading
        dx = self.simulator.target_x - self.simulator.ball.x
        dy = self.simulator.target_y - self.simulator.ball.y

        # Direction is meaningless if we're already at the target
        if (dx != 0) or (dy != 0):

            # Set the magnitude
            vel = vector.set_length([dx, dy, 0.0], speed)

            # Rotate by direction around Z-axis at ball position
            rot = matrix33.create_from_axis_rotation([0.0, 0.0, 1.0], direction)
            vel = matrix33.apply_to_vector(rot, vel)

            # Unpack into ball velocity
            self.simulator.ball_vel.x = vel[0]
            self.simulator.ball_vel.y = vel[1]
            self.simulator.ball_vel.z = vel[2]

    def episode_start(self, config: Dict = None) -> None:
        """Initialize simulator environment using scenario paramters from Inkling.
        
        Parameters
        -------
        config : Dict, optional
            by default None
        """

        # Return to pre-determined good state to avoid accidental episode-episode dependencies
        self.simulator.reset()

        if config is None:
            self.sim_config = {}  
        else:
            self.sim_config = config      
        # Initial control state which are all unitless in [-1..1] 
        self.simulator.roll = self.sim_config.get("initial_roll", self.simulator.roll)
        self.simulator.pitch = self.sim_config.get("initial_pitch", self.simulator.pitch)
        self.simulator.height_z = self.sim_config.get("initial_height_z", self.simulator.height_z)

        # Constants, SI units
        self.simulator.time_delta = self.sim_config.get("time_delta", self.simulator.time_delta)
        self.simulator.jitter = self.sim_config.get("jitter", self.simulator.jitter)
        self.simulator.gravity = self.sim_config.get("gravity", self.simulator.gravity)
        self.simulator.plate_theta_vel_limit = self.sim_config.get(
            "plate_theta_vel_limit", self.simulator.plate_theta_vel_limit
        )
        self.simulator.plate_theta_acc = self.sim_config.get(
            "plate_theta_acc", self.simulator.plate_theta_acc
        )
        self.simulator.plate_theta_limit = self.sim_config.get(
            "plate_theta_limit", self.simulator.plate_theta_limit
        )
        self.simulator.plate_z_limit = self.sim_config.get("plate_z_limit", self.simulator.plate_z_limit)

        self.simulator.ball_mass = self.sim_config.get("ball_mass", self.simulator.ball_mass)
        self.simulator.ball_radius = self.sim_config.get("ball_radius", self.simulator.ball_radius)
        self.simulator.ball_shell = self.sim_config.get("ball_shell", self.simulator.ball_shell)

        self.simulator.obstacle_radius = self.sim_config.get(
            "obstacle_radius", self.simulator.obstacle_radius
        )
        self.simulator.obstacle_x = self.sim_config.get("obstacle_x", self.simulator.obstacle_x)
        self.simulator.obstacle_y = self.sim_config.get("obstacle_y", self.simulator.obstacle_y)

        # A target position the AI can try and move the ball to
        self.simulator.target_x = self.sim_config.get("target_x", self.simulator.target_x)
        self.simulator.target_y = self.sim_config.get("target_y", self.simulator.target_y)

        # Observation config
        self.simulator.ball_noise = self.sim_config.get("ball_noise", self.simulator.ball_noise)
        self.simulator.plate_noise = self.sim_config.get("plate_noise", self.simulator.plate_noise)

        # Update the initial plate metrics from the constants and the controls
        self.simulator.update_plate(plate_reset=True)

        # Initial ball state after updating plate
        self.simulator.set_initial_ball(
            self.sim_config.get("initial_x", self.simulator.ball.x),
            self.sim_config.get("initial_y", self.simulator.ball.y),
            self.sim_config.get("initial_z", self.simulator.ball.z),
        )

        # Velocity set as a vector
        self.simulator.ball_vel.x = self.sim_config.get("initial_vel_x", self.simulator.ball_vel.x)
        self.simulator.ball_vel.y = self.sim_config.get("initial_vel_y", self.simulator.ball_vel.y)
        self.simulator.ball_vel.z = self.sim_config.get("initial_vel_z", self.simulator.ball_vel.z)

        # Velocity set as a speed/direction towards target
        initial_speed = self.sim_config.get("initial_speed", None)
        initial_direction = self.sim_config.get("initial_direction", None)
        if initial_speed is not None and initial_direction is not None:
            self._set_velocity_for_speed_and_direction(initial_speed, initial_direction)


    def episode_step(self, action: Dict):
        """Step through the environment for a single iteration.
        
        Parameters
        ----------
        action : Dict
            An action to take to modulate environment.
        """
        # Use new syntax or fall back to old parameter names
        if action.get('concept_index') == 1: # selector
            action = self.C1.get_action(self.get_state())
        else:
            action = self.C2.get_action(self.get_state())
        
        # Clamp inputs to legal ranges
        self.simulator.roll = self.clamp(
            action.get("input_roll", self.simulator.roll), -1.0, 1.0)
        self.simulator.pitch = self.clamp(
            action.get("input_pitch", self.simulator.pitch), -1.0, 1.0)
        self.simulator.height_z = self.clamp(
            action.get("input_height_z", self.simulator.height_z), -1.0, 1.0)

        self.simulator.step()

    def halted(self) -> bool:
        """Halt current episode if the simulator has reached an unexpected state (if the ball is off the plate).
        
        Returns
        -------
        bool
            Whether to terminate current episode
        """
        return self.simulator.halted()   

    def random_policy(self, state: Dict = None) -> Dict:

        return random_policy(state)         

def env_setup():
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(verbose=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_exists = os.path.exists(".env")
    if not env_file_exists:
        open(".env", "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(".env", "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(".env", "SIM_ACCESS_KEY", access_key)

    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def test_random_policy(
    num_episodes: int = 10,
    num_iterations: int = 5,
):
    """Test a policy using random actions over a fixed number of episodes
    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """
    sim = TemplateSimulatorSession()
    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        sim_state = sim.episode_start()
        while not terminal:
            action = sim.random_policy(sim_state)
            sim.episode_step(action)
            sim_state = sim.get_state()
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations: {sim_state}")
            iteration += 1
            terminal = iteration >= num_iterations

    return sim



def main():
    """Main entrypoint for running simulator connections
    """
    # workspace environment variables
    env_setup()
    load_dotenv(verbose=True, override=True)

    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession()

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # # Load json file as simulator integration config type file
    # with open("moab_interface.json") as file:
    # interface = json.load(file)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=60,
        simulator_context=config_client.simulator_context,
    )
    registered_session = client.session.create(
        workspace_name=config_client.workspace, body=registration_info
    )
    print("Registered simulator.")
    sequence_id = 1
    episode = 0
    iteration = 0

    try:
        while True:
            # Advance by the new state depending on the event type
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            event = client.session.advance(
                workspace_name=config_client.workspace,
                session_id=registered_session.session_id,
                body=sim_state,
            )
            sequence_id = event.sequence_id
            print("[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type))

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1
            elif event.type == "EpisodeStep":
                iteration += 1
                sim.episode_step(event.episode_step.action)
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 0
            elif event.type == "Unregister":
                client.session.delete(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                )
                print("Unregistered simulator.")
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":
    main()
    #test_random_policy()