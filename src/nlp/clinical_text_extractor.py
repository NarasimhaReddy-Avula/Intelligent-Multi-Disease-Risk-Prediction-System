"""
Clinical Text Feature Extraction Pipeline
Extracts lifestyle and history features from patient-provided text using:
1. Rule-based pattern matching (regex)
2. PubMedBERT for semantic understanding
3. Clinical NER for entity extraction
"""

import re
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# NOTE: Install required packages
# pip install transformers torch spacy
# python -m spacy download en_core_web_sm


@dataclass
class LifestyleFeatures:
    """Structured output from clinical text"""
    smoking_status: str  # never, former, current
    smoking_frequency: Optional[str] = None
    alcohol_status: str = "unknown"  # never, occasional, regular, heavy
    alcohol_frequency: Optional[str] = None
    physical_activity: str = "unknown"  # sedentary, low, moderate, high
    family_history: List[str] = None
    symptoms: List[str] = None
    comorbidities: List[str] = None
    medications: List[str] = None
    
    def __post_init__(self):
        if self.family_history is None:
            self.family_history = []
        if self.symptoms is None:
            self.symptoms = []
        if self.comorbidities is None:
            self.comorbidities = []
        if self.medications is None:
            self.medications = []


class ClinicalTextExtractor:
    """
    Extract clinical features from patient history text
    Uses hybrid approach: regex patterns + PubMedBERT embeddings
    """
    
    def __init__(self, use_bert: bool = True):
        """
        Args:
            use_bert: Whether to use PubMedBERT (requires GPU/significant RAM)
        """
        self.use_bert = use_bert
        
        # Smoking patterns
        self.smoking_patterns = {
            'current': [
                r'i\s+smoke',
                r'(i\s+am|i\'m)\s+a\s+smoker',
                r'smoking\s+(\d+)',
                r'(\d+)\s*(cigarettes?|packs?)',
                r'(a|one)\s+packs?\s+(a|per)\s+day',
                r'smok(ing|es?)\s+(about\s+)?(a|one|\d+)\s+packs?',
                r'current\s+smoker',
                r'\b(smoker|smokes|smoking)\b'
            ],
            'former': [
                r'(quit|stopped|gave\s+up)\s+(smoking|cigarettes?|tobacco)',
                r'used\s+to\s+smok',
                r'former\s+smoker',
                r'ex-?smoker',
                r'smok(ed|ing)?\s+(until|till)\s+\d+'
            ],
            'never': [
                r'non-?smoker',
                r'never\s+smoked',
                r'don\'t\s+smoke'
            ]
        }
        
        # Alcohol patterns
        self.alcohol_patterns = {
            'heavy': [
                r'(\d+)[+-]?\s*(beers?|drinks?|glasses?)\s+(daily|per\s+day|every\s+day)',
                r'heavy\s+drink',
                r'drinks?\s+(daily|every\s+day|heavily|a\s+lot|too\s+much)',
                r'drinks?\s+heavily',
                r'alcoholic'
            ],
            'regular': [
                r'drink\s+alcohol\s+(regularly|often)',
                r'drinks?\s+(regularly|often|most\s+(days?|nights?))',
                r'(social|regular)\s+drink'
            ],
            'occasional': [
                r'(occasional|sometimes)\s+drink',
                r'drinks?\s+(on|during|at|over|only\s+on)\s+(the\s+)?weekends?',
                r'(\d+)[-–]?\s*\d*\s*(beers?|drinks?|glasses?)\s+(on|during|at)\s+(the\s+)?weekends?',
                r'weekend\s+drink',
                r'weekends?\s+only',
                r'socially',
                r'drinks?\s+occasionally'
            ],
            'never': [
                r'don\'t\s+drink',
                r'non-?drinker',
                r'never\s+drink'
            ]
        }
        
        # Physical activity patterns
        self.activity_patterns = {
            'sedentary': [
                r'sedentary',
                r'no\s+exercise',
                r'desk\s+job',
                r'desk[-\s]?based',
                r'rarely\s+exercise',
                r'exercise\s+rarely',
                r'barely\s+exercise',
                r'(don\'t|do\s+not)\s+exercise',
                r'no\s+physical\s+activity'
            ],
            'moderate': [
                r'exercise\s+(\d+)\s+times\s+weekly',
                r'moderately\s+active',
                r'walk\s+regularly'
            ],
            'high': [
                r'exercise\s+daily',
                r'very\s+active',
                r'athlete'
            ]
        }
        
        # Disease keywords for family history
        self.disease_keywords = {
            'diabetes': ['diabetes', 'diabetic'],
            'cardiovascular_disease': ['heart attack', 'heart disease', 'bypass', 'stroke'],
            'hypertension': ['high blood pressure', 'hypertension'],
            'chronic_kidney_disease': ['kidney failure', 'kidney disease', 'dialysis'],
            'liver_disease': ['cirrhosis', 'hepatitis', 'liver disease']
        }
        
        # Family member keywords
        self.family_keywords = ['father', 'mother', 'brother', 'sister', 'sibling', 
                                'parent', 'grandmother', 'grandfather', 'family history']
        
        if self.use_bert:
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
                )
                self.model = AutoModel.from_pretrained(
                    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
                )
                print("✅ PubMedBERT loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load PubMedBERT: {e}")
                print("   Falling back to rule-based extraction only")
                self.use_bert = False
    
    def extract_features(self, text: str) -> LifestyleFeatures:
        """
        Main extraction method
        
        Args:
            text: Patient history text
            
        Returns:
            LifestyleFeatures object with extracted information
        """
        text_lower = text.lower()
        
        features = LifestyleFeatures(
            smoking_status=self._extract_smoking(text_lower),
            alcohol_status=self._extract_alcohol(text_lower),
            physical_activity=self._extract_activity(text_lower),
            family_history=self._extract_family_history(text_lower),
            symptoms=self._extract_symptoms(text_lower)
        )
        
        # Extract smoking frequency if current smoker
        if features.smoking_status == "current":
            features.smoking_frequency = self._extract_smoking_frequency(text_lower)
            if features.smoking_frequency is None and re.search(r'(a|one)\s+pack\s+(a|per)\s+day', text_lower):
                features.smoking_frequency = "1 pack/day"
        
        # Extract alcohol frequency if drinker
        if features.alcohol_status in ["occasional", "regular", "heavy"]:
            features.alcohol_frequency = self._extract_alcohol_frequency(text_lower)
            if features.alcohol_frequency is None and re.search(r'weekends?', text_lower):
                features.alcohol_frequency = "weekends"
        
        return features
    
    def _extract_smoking(self, text: str) -> str:
        """Extract smoking status — never/former take precedence over current."""
        for status in ("never", "former", "current"):
            for pattern in self.smoking_patterns.get(status, []):
                if re.search(pattern, text, re.IGNORECASE):
                    return status
        return "unknown"
    
    def _extract_smoking_frequency(self, text: str) -> Optional[str]:
        """Extract how much they smoke"""
        # Look for "X cigarettes per day"
        match = re.search(r'(\d+)\s*(cigarettes?|packs?)\s*(per\s+day|daily|\/day)?', text)
        if match:
            return f"{match.group(1)} {match.group(2)}/day"
        return None
    
    def _extract_alcohol(self, text: str) -> str:
        """Extract alcohol status — never/heavy take precedence (more specific first)."""
        for status in ("never", "heavy", "regular", "occasional"):
            for pattern in self.alcohol_patterns.get(status, []):
                if re.search(pattern, text, re.IGNORECASE):
                    return status
        return "unknown"
    
    def _extract_alcohol_frequency(self, text: str) -> Optional[str]:
        """Extract drinking frequency"""
        match = re.search(r'(\d+)[+-]?\s*(beers?|drinks?|glasses?)', text)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None
    
    def _extract_activity(self, text: str) -> str:
        """Extract physical activity level"""
        for level, patterns in self.activity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        return "unknown"
    
    def _extract_family_history(self, text: str) -> List[str]:
        """Extract diseases mentioned with family members"""
        diseases_found = []
        
        # Check if text mentions family
        has_family_mention = any(kw in text for kw in self.family_keywords)
        
        if has_family_mention:
            # Look for disease keywords near family keywords
            for disease, keywords in self.disease_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        diseases_found.append(disease)
                        break
        
        return list(set(diseases_found))  # Remove duplicates
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract mentioned symptoms"""
        symptom_keywords = {
            'chest_pain': ['chest pain', 'chest discomfort'],
            'fatigue': ['fatigue', 'tired', 'exhausted'],
            'shortness_of_breath': ['shortness of breath', 'breathless'],
            'weight_loss': ['weight loss', 'losing weight'],
            'frequent_urination': ['frequent urination', 'polyuria', 'urinate frequently',
                                   'urinating frequently', 'urinate a lot', 'urinate at night'],
            'excessive_thirst': ['excessive thirst', 'polydipsia', 'thirsty', 'very thirsty'],
            'jaundice': ['yellow', 'jaundice'],
            'edema': ['swelling', 'edema']
        }
        
        symptoms = []
        for symptom, keywords in symptom_keywords.items():
            if any(kw in text for kw in keywords):
                symptoms.append(symptom)
        
        return symptoms
    
    def extract_batch(self, texts: List[str]) -> List[LifestyleFeatures]:
        """Process multiple texts"""
        return [self.extract_features(text) for text in texts]
    
    def to_dict(self, features: LifestyleFeatures) -> Dict:
        """Convert features to dictionary for model input"""
        return {
            'smoking_current': 1 if features.smoking_status == 'current' else 0,
            'smoking_former': 1 if features.smoking_status == 'former' else 0,
            'alcohol_regular': 1 if features.alcohol_status in ['regular', 'heavy'] else 0,
            'physical_activity_sedentary': 1 if features.physical_activity == 'sedentary' else 0,
            'physical_activity_moderate': 1 if features.physical_activity == 'moderate' else 0,
            'physical_activity_high': 1 if features.physical_activity == 'high' else 0,
            'family_history_diabetes': 1 if 'diabetes' in features.family_history else 0,
            'family_history_heart': 1 if 'cardiovascular_disease' in features.family_history else 0,
            'family_history_kidney': 1 if 'chronic_kidney_disease' in features.family_history else 0,
            'family_history_liver': 1 if 'liver_disease' in features.family_history else 0
        }


def clinical_context_notes(features: "LifestyleFeatures") -> List[Dict[str, Any]]:
    """Map extracted lifestyle factors to conservative clinical context notes.

    These are rule-based heuristics shown alongside model scores — they flag
    known risk factors that are NOT among the model's 24 features, so the
    affected score may be conservative. Not diagnostic.

    Returns list of {"factor": str, "targets": [disease targets], "note": str}.
    """
    notes: List[Dict[str, Any]] = []
    if features is None:
        return notes

    if features.smoking_status == "current":
        notes.append({
            "factor": "smoking (current)",
            "targets": ["heart_disease_risk", "kidney_disease_risk"],
            "note": "Smoking damages blood vessels and raises heart and kidney risk — "
                    "quitting is one of the most effective steps to protect both.",
        })
    elif features.smoking_status == "former":
        notes.append({
            "factor": "smoking (former)",
            "targets": ["heart_disease_risk"],
            "note": "Past smoking still leaves some heart risk — staying tobacco-free "
                    "keeps protecting you.",
        })

    if features.alcohol_status in ("regular", "heavy"):
        notes.append({
            "factor": f"alcohol ({features.alcohol_status})",
            "targets": ["liver_disease_risk", "heart_disease_risk"],
            "note": "Regular alcohol use strains the liver and raises blood pressure — "
                    "cutting down protects both liver and heart.",
        })

    if features.physical_activity == "sedentary":
        notes.append({
            "factor": "sedentary lifestyle",
            "targets": ["diabetes_risk", "heart_disease_risk"],
            "note": "Low physical activity raises blood sugar and heart risk — "
                    "even 30 minutes of daily walking helps.",
        })

    fam = set(features.family_history or [])
    fam_map = {
        "diabetes": ("diabetes_risk", "family history of diabetes",
                     "Family history of diabetes raises your personal risk — periodic "
                     "blood sugar checks are recommended."),
        "cardiovascular_disease": ("heart_disease_risk", "family history of cardiovascular disease",
                                   "Family history of heart disease raises your personal risk — "
                                   "keep BP and cholesterol monitored regularly."),
        "chronic_kidney_disease": ("kidney_disease_risk", "family history of kidney disease",
                                   "Family history of kidney disease raises your personal risk — "
                                   "periodic kidney function tests are wise."),
        "liver_disease": ("liver_disease_risk", "family history of liver disease",
                          "Family history of liver disease raises your personal risk — "
                          "periodic liver function tests are wise."),
    }
    for key, (target, label, note) in fam_map.items():
        if key in fam:
            notes.append({"factor": label, "targets": [target], "note": note})

    return notes


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = ClinicalTextExtractor(use_bert=False)  # Set True if you have PubMedBERT installed
    
    # Test examples
    test_cases = [
        "I smoke about 10 cigarettes per day, drink alcohol on weekends, and have a sedentary desk job. My father had a heart attack.",
        "Non-smoker, exercise 3 times weekly, father and brother have diabetes",
        "I drink 4-5 beers daily for 20 years. Recently noticed yellowing of eyes and fatigue.",
        "I've had type 2 diabetes for 15 years. My father had kidney failure. I smoke 15 cigarettes daily."
    ]
    
    print("=" * 80)
    print("CLINICAL TEXT FEATURE EXTRACTION DEMO")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Case {i}:")
        print(f"Text: {text}\n")
        
        features = extractor.extract_features(text)
        
        print("Extracted Features:")
        print(f"  🚬 Smoking: {features.smoking_status}", end="")
        if features.smoking_frequency:
            print(f" ({features.smoking_frequency})")
        else:
            print()
        
        print(f"  🍺 Alcohol: {features.alcohol_status}", end="")
        if features.alcohol_frequency:
            print(f" ({features.alcohol_frequency})")
        else:
            print()
        
        print(f"  🏃 Activity: {features.physical_activity}")
        
        if features.family_history:
            print(f"  👨‍👩‍👦 Family History: {', '.join(features.family_history)}")
        
        if features.symptoms:
            print(f"  🩺 Symptoms: {', '.join(features.symptoms)}")
        
        # Show as feature vector for model
        feature_dict = extractor.to_dict(features)
        print(f"\n  📊 Feature Vector (for ML model):")
        print(f"     {feature_dict}")
        
        print("-" * 80)
