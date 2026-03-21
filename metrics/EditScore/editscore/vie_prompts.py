# This file is generated automatically through parse_prompt.py
_context_no_delimit_reasoning_first = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning very concise and short.):
{
"reasoning" : "...",
"score" : [...]
}
"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
Both the original image and the edited image are masked images since the image contains multiple objects and we want you to only focus on the intended object.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_tie_rule_SC = """
From a scale 0 to 10:
A score from 0 to 10 will be given based on the success of the editing.
- 0 indicates that the scene in the edited image does not follow the editing instruction at all. 
- 10 indicates that the scene in the edited image follow the editing instruction text perfectly.
Score1 ONLY evaluates whether the instruction-required modification is correctly executed on the intended target, regardless of any additional changes or visual quality.

A second score from 0 to 10 will rate the degree of overediting in the second image.
- 0 indicates that the scene in the edited image contains any unintended modification beyond the instruction or is completely different from the original.
- 10 indicates that only the modifications explicitly required by the instruction are applied, with no additional changes.
Score2 ONLY evaluates whether any object or attribute not mentioned in the instruction is modified; visual quality, realism, shading, lighting, or texture differences must not affect the second score unless they introduce a new object or attribute change.

Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

# # 原图 + 编辑图 + 编辑指令
# _prompts_0shot_rule_PQ = """RULES:
# Two full images will be provided: the first is the original image, and the second is the edited image.
# The objective is to evaluate the visual quality of the edited image (the second image), using the original image (the first image) as reference.
# The Editing instruction will be provided only as reference for intended changes.
# Do NOT evaluate instruction-following success, editing errors, or overediting; ONLY assess the visual quality of the edited image (the second image).
# Do NOT treat instruction-required modifications as violations of physical laws or common-sense realism.

# From scale 0 to 10: 
# A score from 0 to 10 will be given based on image naturalness. 
# (
#     0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
#     10 indicates that the image looks natural.
# )
# A second score from 0 to 10 will rate the image artifacts. 
# (
#     0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or ghosting artifacts, or blurred faces, or unusual body parts, or subjects not harmonized. 
#     10 indicates the image has no artifacts.
# )
# Put the score in a list such that output score = [naturalness, artifacts]

# Editing instruction: <instruction>
# """

# # # 原图 + 编辑图
# _prompts_0shot_rule_PQ = """RULES:
# Two full images will be provided: the first is the original image, and the second is the edited image.
# The objective is to evaluate the visual quality of the edited image (the second image), using the original image (the first image) as reference.
# Do NOT consider the presence of newly introduced or modified elements as unnatural unless they cause visible rendering inconsistencies or artifacts.
# Naturalness must be judged only by rendering coherence and NOT by whether objects or scenes appear artificial, unusual, or semantically implausible.

# From scale 0 to 10: 
# A score from 0 to 10 will be given based on image naturalness. 
# (
#     0 indicates severe rendering inconsistency (e.g., incorrect lighting/shadows, broken geometry/perspective, texture collapse, or poor blending/compositing).
#     10 indicates fully coherent rendering with consistent lighting/shadows, geometry/perspective, texture quality, and blending/compositing in the edited image.
# )
# A second score from 0 to 10 will rate the image artifacts. 
# (
#     0 indicates that the image contains a large portion of distortion, or ghosting artifacts, or blurred faces, or unusual body parts, or subjects not harmonized. 
#     10 indicates the image has no artifacts.
# )
# Put the score in a list such that output score = [naturalness, artifacts]
# """

# 原图 + 编辑图 MistralAI
# _prompts_0shot_rule_PQ = """
# Two images will be provided: The first being the original image and the second being an AI edited version of the first. So you may not worry about privacy or confidentiality.

# You must focus solely on the technical rendering quality and visible low-level artifacts in the edited image.
# Do NOT consider whether the scene is realistic, physically plausible, stylistically unusual, or semantically natural.
# Severe blur, ghosting, double exposure, or motion-smear — especially on faces or primary subjects — should be treated as major rendering failures.
# Do NOT penalize changes that are clearly visible but cleanly rendered (e.g., color changes, object replacement, attribute modifications), as long as there are no rendering artifacts.

# Your evaluation should focus only on:
# - Distortions
# - Deformed anatomy or broken geometry
# - Blur, ghosting, or duplicated edges
# - Warped or corrupted textures
# - Blending or compositing artifacts

# Rate the edited image on a scale from 0 to 10, where:
# - 0 indicates severe visual artifacts.
# - 10 indicates an artifact-free image.
# """

# 原图 + 编辑图 GPT
_prompts_0shot_rule_PQ = """RULES:
Two images are provided:
- Image 1: original image
- Image 2: an edited version of Image 1

You must focus solely on the technical quality and artifacts in the edited image (Image 2), using Image 1 as reference, and **do not consider whether the context is natural or not**.

Your evaluation should focus on:
- Distortions
- Unusual body parts or proportions
- Unnatural Object Shapes

Rate the edited image on a scale from 0 to 10, where:
- 0 indicates significant AI-artifacts.
- 10 indicates an artifact-free image.
"""

# Do NOT evaluate semantic plausibility, realism, typical appearance, or unusual colors/materials/styles if cleanly rendered.
# Changes compared to Image 1 must NOT be penalized unless they introduce rendering defects.
# Your evaluation should focus on:
# - Blur, ghosting, double edges
# - Distortions, or scratches
# - Visible pixel-level artifacts

# # Active PQ prompt: edited image + editing instruction
# _prompts_0shot_rule_PQ = """RULES:
# The image is an AI-generated image.
# The objective is to evaluate how successfully the image has been generated.
# An editing instruction will be provided only as reference for intended changes, so do NOT treat instruction-required changes as violation of common-sense realism.
# Do NOT evaluate instruction-following success, editing errors, or overediting; ONLY assess the visual quality of the image.
# Before assigning scores, you MUST explicitly check for local rendering defects such as ghosting, blending errors, boundary halos, inconsistent focus or missing contact shadows. If such defects exist, they must reduce the artifact score.

# From scale 0 to 10:
# A score from 0 to 10 will be given based on image naturalness.
# (
#     0 indicates that the scene in the image does not look natural at all or gives an unnatural feeling such as wrong sense of distance, wrong shadow, or wrong lighting.
#     10 indicates that the image looks natural.
# )
# A second score from 0 to 10 will rate the image artifacts.
# (
#     0 indicates that the image contains a large portion of distortion, watermark, scratches, ghosting artifacts, blurred faces, unusual body parts, or subjects not harmonized.
#     10 indicates the image has no artifacts.
# )
# Put the score in a list such that output score = [naturalness, artifacts]

# Editing instruction: <instruction>
# """

# 仅编辑图
# _prompts_0shot_rule_PQ = """RULES:

# The image is an AI-generated image.
# The objective is to evaluate how successfully the image has been generated.

# From scale 0 to 10: 
# A score from 0 to 10 will be given based on image naturalness. 
# (
#     0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
#     10 indicates that the image looks natural.
# )
# A second score from 0 to 10 will rate the image artifacts. 
# (
#     0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
#     10 indicates the image has no artifacts.
# )
# Put the score in a list such that output score = [naturalness, artifacts]
# """

# # # 编辑图
# _prompts_0shot_rule_PQ = """RULES:
# The image is an AI-edited image.
# The objective is to evaluate how successfully the image has been edited.
# You must focus solely on the technical quality and artifacts in the edited image, and **do not consider whether the context is natural or not**.
# Do NOT judge based on typical appearance, or whether the object matches common knowledge; ONLY assess the visual quality of the edited image

# Your evaluation should focus on:
# - Distortions
# - Blurriness, ghosting, or unnatural texture transitions
# - Unusual body parts or proportions
# - Geometric deformation, broken topology, duplicated limbs, warped edges
# - Inconsistent lighting or color blending artifacts

# Rate the edited image on a scale from 0 to 10, where:
# - 0 indicates significant AI-artifacts.
# - 10 indicates an artifact-free image.
# """