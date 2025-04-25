import gym
import d4rl # Make sure d4rl is installed and imported to register the environments

# Define the environment name
env_name = 'maze2d-umaze-v1'

try:
    # Create the environment instance
    env = gym.make(env_name)

    # Access the reference scores
    # D4RL typically stores these directly on the environment object
    # or within its spec/metadata after loading.
    # Common attributes are ref_min_score and ref_max_score
    if hasattr(env, 'ref_min_score') and hasattr(env, 'ref_max_score'):
        random_score = env.ref_min_score
        expert_score = env.ref_max_score

        print(f"Reference scores for environment: {env_name}")
        print(f"  Random Score (ref_min_score): {random_score}")
        print(f"  Expert Score (ref_max_score): {expert_score}")
    elif hasattr(env.spec, 'kwargs') and 'ref_min_score' in env.spec.kwargs and 'ref_max_score' in env.spec.kwargs:
        # Sometimes scores might be in the environment spec kwargs
        random_score = env.spec.kwargs['ref_min_score']
        expert_score = env.spec.kwargs['ref_max_score']
        print(f"Reference scores for environment: {env_name}")
        print(f"  Random Score (ref_min_score): {random_score}")
        print(f"  Expert Score (ref_max_score): {expert_score}")
    else:
        # If the attributes aren't found directly, they might be accessible differently
        # depending on the exact d4rl version or setup.
        # The get_normalized_score function implies these values exist somewhere.
        # As a fallback, try accessing the internal info dict if available (less standard)
        try:
            # Attempt to access the internal infos dict (may vary by version)
            from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE
            random_score = REF_MIN_SCORE.get(env_name)
            expert_score = REF_MAX_SCORE.get(env_name)
            if random_score is not None and expert_score is not None:
                 print(f"Reference scores for environment: {env_name} (from internal infos)")
                 print(f"  Random Score: {random_score}")
                 print(f"  Expert Score: {expert_score}")
            else:
                 print(f"Could not find reference scores directly attached to the '{env_name}' env object or in standard internal infos.")
                 print("You might need to check the specific version of d4rl or its source code (e.g., d4rl/infos.py) for exact values.")
        except ImportError:
             print(f"Could not find reference scores directly attached to the '{env_name}' env object.")
             print("Could not import reference scores from d4rl.infos.")
             print("You might need to check the specific version of d4rl or its source code (e.g., d4rl/infos.py) for exact values.")

except gym.error.Error as e:
    print(f"Error creating environment '{env_name}': {e}")
    print("Please ensure 'd4rl' is correctly installed and the environment name is valid.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")