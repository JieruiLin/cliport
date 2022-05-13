import numpy as np
import clip 

device = "cuda" if torch.cuda.is_available() else "cpu"
# "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
# only use train colors
color_names = ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']
packing_lang_template = "pack the {obj} block in the brown box"
stacking_lang_template = "stack the {pick} block on {place}"
put_lang_template = "put the {pick} block in a {place} bowl"

all_languages = []
for i in range(len(color_names)):
    all_languages.append(packing_lang_template.format(obj=color_names[i]))
    all_languages.append(stacking_lang_template.format(pick=color_names[i], place="the lightest brown block"))
    all_languages.append(stacking_lang_template.format(pick=color_names[i], place="the middle brown block"))
    for j in range(len(color_names)):
        if i != j:
            all_languages.append(put_lang_template.format(pick=color_names[i], place=color_names[j]))

        for k in range(len(color_names)):
            if i != j and i != k and j != k:
                all_languages.append(stacking_lang_template.format(pick=color_names[i], place=f"the {color_names[j]} and {color_names[k]} blocks"))

all_tokens = clip.tokenize(all_languages).to(device)
all_actions = clip_model.encode_text(all_tokens)
np.save('/home/huihanl/cliport/data/language_dictionary.npy', all_languages)
np.save('/home/huihanl/cliport/data/action_dictionary.npy', all_actions.detach().cpu().numpy())