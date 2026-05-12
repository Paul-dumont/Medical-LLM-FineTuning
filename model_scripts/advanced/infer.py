#!/usr/bin/env python3
"""
Standalone inference script supporting multiple medical LLM models
No local dependencies required - works with just pip install
"""

import json
from pathlib import Path
import urllib.request
import sys

# ============================================================================
# AVAILABLE MODELS - Choose one below
# ============================================================================
AVAILABLE_MODELS = {
    # "phi_ortho": {
    #     "repo_id": "dcbia/Phi-3.5-Mini-Instruct-Ortho",
    #     "file_name": "model-q4_0.gguf",
    #     "local_name": "Phi-3.5-Mini-Ortho.gguf",
    #     "description": "Phi 3.5 Mini Ortho (2.4 GB)",
    #     "type": "ortho"
    # },
    # "llama_ortho": {
    #     "repo_id": "dcbia/Meta-Llama-3.1-8B-Instruct-Ortho",
    #     "file_name": "model-q4_0.gguf",
    #     "local_name": "Meta-Llama-3.1-8B-Ortho.gguf",
    #     "description": "Meta Llama 3.1 8B Ortho (4.7 GB)",
    #     "type": "ortho"
    # },
    "qwen_tmj_mini": {
        "repo_id": "dcbia/Qwen-2.5-1.5B-Instruct-TMJ",
        "file_name": "Qwen-2.5-1.5B-Instruct-TMJ-q4_0.gguf",
        "local_name": "Qwen-2.5-1.5B-TMJ.gguf",
        "description": "Qwen 2.5 1.5B TMJ (1 GB)",
        "type": "tmj"
    },
    "qwen_tmj_max": {
        "repo_id": "dcbia/Qwen-2.5-7B-Instruct-TMJ",
        "file_name": "Qwen-2.5-7B-Instruct-TMJ-q4_0.gguf",
        "local_name": "Qwen-2.5-7B-TMJ.gguf",
        "description": "Qwen 2.5 7B TMJ (4.4 GB)",
        "type": "tmj"
    }
}

# ============================================================================
# CONFIGURATION - Choose which model to use
# ============================================================================
SELECTED_MODELS = list(AVAILABLE_MODELS.keys())  # Run all models - Change to ["llama_ortho"] for single model
# ============================================================================

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)

# Setup cache directory
project_root = Path.home() / ".cache" / "medical-llm"
project_root.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MEDICAL LLM INFERENCE - TESTING ALL SELECTED MODELS")
print("=" * 80)
print(f"Models to test: {', '.join(SELECTED_MODELS)}")
print()

# Store results
results = {}

# Loop through all selected models
for model_key in SELECTED_MODELS:
    if model_key not in AVAILABLE_MODELS:
        print(f"Error: Unknown model '{model_key}'")
        continue
    
    model_config = AVAILABLE_MODELS[model_key]
    model_path = str(project_root / model_config["local_name"])
    
    print("\n" + "=" * 80)
    print(f"MODEL: {model_config['description']}")
    print("=" * 80)
    
    # Download model if not cached
    if not Path(model_path).exists():
        print(f"\n📥 DOWNLOADING FROM HUGGING FACE...")
        modelUrl = f"https://huggingface.co/{model_config['repo_id']}/resolve/main/{model_config['file_name']}"
        print(f"Repo: {model_config['repo_id']}/{model_config['file_name']}")
        print(f"Path: {model_path}")
        print()
        
        try:
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, int(downloaded / total_size * 100))
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
            
            urllib.request.urlretrieve(modelUrl, model_path, progress_hook)
            print("\n✅ Model downloaded successfully!")
        except Exception as e:
            print(f"\n❌ Error downloading model: {e}")
            continue
    else:
        print(f"✅ Model already cached")
    
    print(f"\n⏳ Loading model into memory...")
    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        continue

    
    # Select clinical note based on model type
    if model_config["type"] == "tmj":
        clinical_note = """B_001 B_001 is 26 years old and referred by Dr. XXX. B_001 had no problems or symptoms prior to an MVA that occurred 4 months ago. She was riding in a bus that got hit by an 18-wheeler from behind and slammed her into the side of the bus on her left side and she hit the whole left side of her face and left side of her body. She rates her headaches at a level of 6 in the frontal, temporal and posterior head area and has particular problems with pain behind her eyes. She rates her TMJ pain at 6, average daily pain at 8, jaw function at 5, diet at 3, disability at 2. She gets frequent left-sided earaches, frequent tinnitus, moderate to frequent vertigo, and she has hearing loss when she gets the tinnitus. Other joints that bother her include her left hip, her knees, back (3 herniated disks) and neck. She had none of these problems prior to her accident. She also has some neurological factors affecting her legs and arms. She is aware of severe clenching and bruxing at nighttime and also during the day. She did not have this prior to her injury. The only medication she takes is Tylenol at times. She has essentially had no specific treatment to date.

        B_001 has good facial symmetry and balance. She has relatively normal upper lip length of about 21 mm and upper tooth lip relationship of 2 mm. She has good balance in her profile. Intraorally, she has all of her lower teeth back through the second molars. She has some crowding in the lower anterior arch and she has adequate attached gingiva. In the upper arch, she has all of her teeth back through the third molars. She does have an impacted mandibular left third molar. She has a broken down mandibular first molar with a broken tooth and restoration. Oropharyngeal tissues look relatively normal except the tonsils are of moderate size. She has basically a Class I occlusion. She has 1 mm anterior over bite relationship. Nasal structures look relatively normal. She apparently occasionally snores and has some moderate daytime tiredness, but is not aware of having overt sleep apnea. She is aware of breathing through her mouth at nighttime. To palpation, she has moderate to high moderate pain throughout the head and neck area. The worse pain to palpation is in the left TMJ. The maximum incisal opening is 49 mm and opening without pain 30 mm. The excursion movement to the right is 5 mm and to the left 6 mm.

        RADIOGRAPHIC EVALUATION
        Panorex shows the presence of all teeth. The mandibular left third molar is in distal angulation and impacted.
        Coronal view shows that she has an enlarged mandibular left turbinate. She has a little sinusitis with some increased thickness of the synovial lining in the floor of the nose.

        In the axial view, she has some increased thickening of the sinus membrane along the posterior and lateral aspect of the sinus. The left turbinate appears significantly enlarged compared to the right side.

        Left TMJ sagittal view shows a condyle that is posteriorly positioned in the fossa, but there is a good anterosuperior and anterior joint space.
        Right TMJ sagittal view shows a condyle that is posteriorly positioned in the fossa, but there is a good anterosuperior and anterior joint space.

        Left TMJ coronal view shows a condyle that has fairly good morphology. The lateral rim of the fossa curves inferiorly about 2 or 3 mm. There appears to be adequate joint space, although slightly increased on the medial side.

        Right TMJ coronal view shows a condyle that has good mediolateral width. The top of the condyle is more flattened as is the fossa.
        Lateral cephalometric radiograph shows that she has a skeletal and occlusal Class I relationship. Oropharyngeal airway measures about 9 mm at the soft palate level and about 7 to 8 mm at the base of the tongue.

        B_001 does state that her speech is affected sometimes by the muscle spasms that she develops. When she is talking to the parents of her school kids, her jaw sometimes will pull off to the side. She sometimes slurs her words because of the muscle dysfunction that she currently has.

        Basic diagnoses for B_001 would be as follows:
        Bilateral TMJ arthritis and articular discs anteriorly displaced with early reduction.
        Mandibular hyperplasia. TMJ pain.
        Headaches. Myofascial pain.
        Impacted mandibular left third molar and malaligned maxillary third molars and mandibular right third molar.

        Recommended treatment would be as follows:
        1.	Medications: Klonopin 1 mg tablet, dispensing 30 tablets, 1 tablet q.h.s., refills x2.
        2.	Surgery. a.	Bilateral TMJ articular disc repositioning and ligament repair with Mitek anchor. b.	Bilateral mandibular ramus sagittal split osteotomies. c.	Application of maxillary and mandibular arch bars. d.	Removal of third molars x4."""
    else:
        clinical_note = """Appliance Adjustment 13y F presents early with grandmother for appliance adjustment. LR5 debonded, that bracket was rebonded.
        Progress photos taken today. Pt is now more class II on the right, very close to class I on the left. Upper midline is off to the left ~2mm.
        Pt reports that she is wearing the elastics full time. OH is good today, pt had cleaning recently. UL6 is now erupting! It is coming in buccally.
        We will plan to take a progress 8x8 and bond 7s once class II is corrected. Pt graduated middle school since last appt. She went to [LOCATION] for vacation. 
        Her grandmother is going to [LOCATION] later this month for her friend's wedding and to celebrate her birthday. She has never been before and is very excited for her trip. 
        U: 17 x 25 SS 6-6, reduced PC 6-6 L: 19 x 25 SS 6-6 Elastics: continue "bear" class II bilateral full time Removed bite turbos on lingual of U1s this appt.
        Worked with [DOCTOR] NV: appliance adjustment, check class II correction and eruption of UL7"""
    
    print(f"\n🚀 PERFORMING INFERENCE ({model_config['type'].upper()})...")
    
    # Format prompt using chat template
    prompt = f"""<|start_header_id|>user<|end_header_id|>

{clinical_note}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    try:
        # Generate response
        response = model(
            prompt,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            echo=False,
        )
        
        output = response["choices"][0]["text"].strip()
        
        print(f"\n📝 OUTPUT:")
        print("-" * 80)
        print(output)
        print("-" * 80)
        
        # Store result
        results[model_key] = {
            "description": model_config["description"],
            "type": model_config["type"],
            "output": output
        }
        
        # Unload model from memory to save RAM
        del model
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        results[model_key] = {
            "description": model_config["description"],
            "type": model_config["type"],
            "output": f"ERROR: {str(e)}"
        }

print("\n\n" + "=" * 80)
print("SUMMARY - ALL MODELS TESTED")
print("=" * 80)
for model_key, result in results.items():
    status = "✅" if not result["output"].startswith("ERROR") else "❌"
    print(f"{status} {result['description']} ({result['type'].upper()})")
print("=" * 80)

