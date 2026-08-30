CONTEXT = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning very concise and short.):
{
"reasoning" : "...",
"score" : [...]
}
"""

SC_BATCH_CONTEXT = """You are a professional digital artist. You will evaluate multiple independent image-edit items in one request.
All input images are AI-generated. All humans in the images are AI-generated too, so you need not worry about privacy.

Treat every edit item independently. For each item, the original masked image is followed immediately by its edited masked image. Do not transfer evidence, scores, or reasoning between items. Return exactly one result for every supplied crop_index using the required response schema.
"""

TWO_IMAGE_EDIT_RULE = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
Both the original image and the edited image are masked images since the image contains multiple objects and we want you to only focus on the intended object.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

SC_RULE = """
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

SC_BATCH_RULE = """RULES:

For every independent edit item, evaluate two scores from 0 to <score_range>.

prompt_following:
- 0 indicates that the edited image does not follow the item's editing instruction at all.
- <score_range> indicates that the instruction-required modification is executed perfectly on the intended target.
- Evaluate only whether the required modification is correctly executed, regardless of additional changes or visual quality.

consistency:
- 0 indicates unintended modification beyond the instruction or a completely different result.
- <score_range> indicates that only modifications explicitly required by the instruction are applied, with no additional changes.
- Evaluate only unintended object or attribute changes. Visual quality, realism, shading, lighting, or texture differences must not affect this score unless they introduce a new object or attribute change.

Use each item's crop_index exactly as supplied. Keep each item's reasoning concise and base it only on that item's instruction and image pair.
"""

PQ_RULE = """RULES:
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
