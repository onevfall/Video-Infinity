

# Configuration
BASE_CONFIG="examples/multi_promts.json"
GPU_IDS="6,7"  # Specify GPUs here

# Array of prompts (each element contains multiple prompts separated by ;)
declare -a prompt_groups=(
    # sunny sky 和 snowy sky
    "The camera follows closely behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road. The rear of the SUV remains in full view, the tires kicking up clouds of dust that trail behind as it moves forward. The towering pine trees on either side of the rugged mountain slope frame the shot, while sunlight filters through the branches, casting a golden glow over the scene. The winding dirt road stretches out ahead, curving gently into the distance, with no other vehicles in sight. The camera stays locked on the back of the SUV, capturing its smooth navigation through the challenging terrain. Above, the sky is clear and vibrant blue, with wispy clouds drifting across, enhancing the sense of adventure as the vehicle climbs the steep path.;\
    The camera follows directly behind a white vintage SUV with a black roof rack as it speeds along a steep snow-covered road. The SUV’s rear is consistently framed at the center of the shot, with snow spraying up from the tires as it powers through the icy terrain. The surrounding pine trees tower on either side, their branches heavy with snow. The snow-covered road winds into the distance, and the camera keeps its focus on the back of the vehicle, capturing the SUV as it expertly maneuvers through the winter landscape. Above, the sky is pale and overcast, with muted sunlight struggling to break through the thick clouds, creating a serene yet chilly atmosphere as the camera tracks the vehicle’s snowy ascent."

    # 颜色变化  "red car" 和 "white car"
    "The camera follows closely behind a red vintage SUV with a black roof rack as it speeds up a steep dirt road. The rear of the SUV remains in full view, the tires kicking up clouds of dust that trail behind as it moves forward. The towering pine trees on either side of the rugged mountain slope frame the shot, while sunlight filters through the branches, casting a golden glow over the scene. The winding dirt road stretches out ahead, curving gently into the distance, with no other vehicles in sight. The camera stays locked on the back of the SUV, capturing its smooth navigation through the challenging terrain. Above, the sky is clear and vibrant blue, with wispy clouds drifting across, enhancing the sense of adventure as the vehicle climbs the steep path.;\
    The camera follows closely behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road. The rear of the SUV remains in full view, the tires kicking up clouds of dust that trail behind as it moves forward. The towering pine trees on either side of the rugged mountain slope frame the shot, while sunlight filters through the branches, casting a golden glow over the scene. The winding dirt road stretches out ahead, curving gently into the distance, with no other vehicles in sight. The camera stays locked on the back of the SUV, capturing its smooth navigation through the challenging terrain. Above, the sky is clear and vibrant blue, with wispy clouds drifting across, enhancing the sense of adventure as the vehicle climbs the steep path."

    # 人流散去  "crowded beach" 和 "empty beach" 
    "A bustling boulevard is packed with people strolling in every direction, their footsteps blending with the hum of city life. The sidewalks are filled with families, tourists, and locals, creating a lively, energetic atmosphere. Cars crawl through the traffic, honking occasionally as bicycles weave between them. The trees lining the boulevard cast soft shadows, but the sunlight breaks through, illuminating the vibrant colors of store signs and street vendors. The sound of conversations, laughter, and occasional music fills the air, as the camera captures the constant motion of the crowded street.;\
    A quiet boulevard, nearly empty of pedestrians, stretches out under the warm sunlight. Only a few people wander along the wide sidewalks, their footsteps echoing softly in the calm air. Cars move freely, with no traffic to slow them down, and the occasional cyclist passes by with ease. The trees lining the boulevard sway gently in the breeze, their shadows stretching across the quiet street. The stores and street vendors stand open but relatively undisturbed, and the peaceful atmosphere is a stark contrast to the usual hustle, as the camera glides along the nearly deserted boulevard."

    "A portrait of an old man, facing the camera in a softly lit room, the fine lines and wrinkles on his weathered face captured in stunning detail. His eyes are deep and thoughtful, reflecting a life filled with experiences. The lighting highlights the texture of his skin and the subtle gray in his neatly groomed hair. His expression is neutral yet dignified, offering a quiet sense of wisdom. The background is blurred to draw full attention to the man’s face, ensuring the focus remains on the sharp, intricate details of his portrait.;\
    A portrait of an old man, facing the camera in a softly lit room, the fine lines and wrinkles on his weathered face captured in stunning detail. His eyes are bright, and a gentle smile lifts the corners of his mouth, revealing a sense of warmth and kindness. The lighting highlights the texture of his skin and the subtle gray in his neatly groomed hair. His smile adds a touch of joy to his dignified appearance. The background is blurred to draw full attention to the man’s smiling face, ensuring the focus remains on the sharp, intricate details of his portrait."

    "A video of a dog standing still in the middle of a sunlit park, its fur gently ruffled by the soft breeze. The dog’s ears are perked up as it attentively observes its surroundings. The sunlight casts a warm glow on its coat, highlighting the texture of its fur. The grass beneath its paws is lush and green, with a few leaves scattered around. The dog remains calm and steady, occasionally blinking or twitching its ears, while the background shows distant trees swaying slightly in the wind. The scene feels peaceful and serene as the camera captures the dog’s poised stance.;\
    A video of a dog running toward the camera through a sunlit park, its fur streaming back with each quick stride. The dog's ears bounce with its energetic movements, and its eyes are focused forward with a playful intensity. Its paws hit the grass rhythmically, sending small patches of dirt and leaves into the air as it closes the distance. The sunlight gleems off the dog's coat, giving it a vibrant shine, highlighting the excitement in its approach. The grass beneath it is lush and green, with a few leaves swirling around in the breeze created by its swift motion. The camera captures the dog’s joyful sprint, its lively energy filling the scene as it eagerly races toward us, leaving a trail of playful motion behind."

    # 天气变化  "cloudy sky" 和 "sunny sky" 
    "A video of the sky, where thick, dense clouds stretch endlessly across the frame as the camera looks upward from the ground. The heavy clouds completely obscure any view of the sun, forming a soft, muted gray blanket over the sky. The layers of clouds create a dim, overcast atmosphere, with no sunlight breaking through. The light is diffused, casting a gentle, dull glow over the scene, enhancing the sense of stillness.;\
    A video of the sky, brilliantly clear and vibrant, with not a single cloud in sight. The camera points upward from the ground, capturing the sun shining brightly at the center of the frame, casting a warm, golden light across the entire scene. The sky is a deep, unbroken blue, stretching endlessly in every direction. The camera pans slowly, emphasizing the vast openness and the crisp, clean atmosphere as the sun radiates in the cloudless sky, filling the scene with brightness and a sense of clarity."
)

# Process each group of prompts
for prompt_group in "${prompt_groups[@]}"; do
    echo "Processing new prompt group"
    echo "Using GPUs: ${GPU_IDS}"
    
    # Pass prompts directly through environment variable
    export PROMPT_GROUP="$prompt_group"
    
    # Run inference with specified GPUs
    python inference.py --config "$BASE_CONFIG" --gpu_ids $GPU_IDS
    
    echo "Waiting 10 seconds before next group..."
    
done