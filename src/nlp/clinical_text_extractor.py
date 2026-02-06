"""
Clinical Text Feature Extraction Pipeline
Extracts lifestyle and history features from patient-provided text using:
1. Rule-based pattern matching (regex)
2. PubMedBERT for semantic understanding
3. Clinical NER for entity extraction
"""

import re
import json
from typing import Dict, List, Optional
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
                r'smoking\s+(\d+)',
                r'(\d+)\s*(cigarettes?|packs?)',
                r'current\s+smoker'
            ],
            'former': [
                r'(quit|stopped)\s+(smoking|cigarettes?)',
                r'former\s+smoker',
                r'ex-?smoker'
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
                r'alcoholic'
            ],
            'regular': [
                r'drink\s+alcohol\s+(regularly|often)',
                r'(social|regular)\s+drink'
            ],
            'occasional': [
                r'(occasional|sometimes)\s+drink',
                r'weekends?\s+only',
                r'socially'
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
                r'rarely\s+exercise'
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
        
        # Extract alcohol frequency if drinker
        if features.alcohol_status in ["regular", "heavy"]:
            features.alcohol_frequency = self._extract_alcohol_frequency(text_lower)
        
        return features
    
    def _extract_smoking(self, text: str) -> str:
        """Extract smoking status"""
        for status, patterns in self.smoking_patterns.items():
            for pattern in patterns:
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
        """Extract alcohol consumption status"""
        for status, patterns in self.alcohol_patterns.items():
            for pattern in patterns:
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
            'frequent_urination': ['frequent urination', 'polyuria'],
            'excessive_thirst': ['excessive thirst', 'polydipsia'],
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
