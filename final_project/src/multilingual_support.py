"""
Multilingual Support System for Physical AI & Humanoid Robotics Textbook
Provides translation, localization, and cultural adaptation for multiple languages including Urdu
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
import json

logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Supported language codes"""
    ENGLISH = "en"
    URDU = "ur"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class TranslationResult:
    """Result of translation operation"""
    original_text: str
    translated_text: str
    source_language: LanguageCode
    target_language: LanguageCode
    confidence: float
    translation_time: float
    quality_score: float


@dataclass
class LocalizationResult:
    """Result of localization operation"""
    localized_content: str
    cultural_adaptations: List[str]
    measurement_units: str
    currency_format: str
    date_format: str


class UrduTextProcessor:
    """Processor for Urdu language text"""

    def __init__(self):
        self.urdu_characters = set([
            'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ',
            'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ہ', 'ۂ', 'ۃ', 'ي', 'ۓ'
        ])

        self.urdu_numerals = {
            '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
        }

        self.western_to_urdu_numerals = {v: k for k, v in self.urdu_numerals.items()}

        self.urdu_punctuation = {
            '،': ',',  # Urdu comma
            '؛': ';',  # Urdu semicolon
            '؟': '?',  # Urdu question mark
            '۔': '.',  # Urdu full stop
        }

        self.right_to_left_punctuation = ['؟', '،', '؛', '۔']

    def is_urdu_text(self, text: str) -> bool:
        """Check if text contains Urdu characters"""
        text_chars = set(text)
        urdu_intersection = text_chars.intersection(self.urdu_characters)
        return len(urdu_intersection) > 0

    def normalize_urdu_text(self, text: str) -> str:
        """Normalize Urdu text for consistent processing"""
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFC', text)

        # Handle common normalization issues
        replacements = {
            'آ': 'ا',  # Normalize Alef with Madda
            'إ': 'ا',  # Normalize Alef with Hamza Below
            'أ': 'ا',  # Normalize Alef with Hamza Above
            'ؤ': 'و',  # Normalize Waw with Hamza
            'ئ': 'ی',  # Normalize Yeh with Hamza
        }

        for wrong, correct in replacements.items():
            normalized = normalized.replace(wrong, correct)

        return normalized

    def convert_numbers(self, text: str, to_urdu: bool = True) -> str:
        """Convert between Western Arabic and Urdu numerals"""
        conversion_map = self.urdu_numerals if to_urdu else self.western_to_urdu_numerals

        result = text
        for western, urdu in conversion_map.items():
            result = result.replace(western, urdu)

        return result

    def handle_bidirectional_text(self, text: str) -> str:
        """Handle bidirectional text mixing Arabic script with Latin"""
        # Add right-to-left embedding for Urdu content
        if self.is_urdu_text(text):
            # Use RLE (Right-to-Left Embedding) and PDF (Pop Directional Format) characters
            return '\u202B' + text + '\u202C'  # RLE and PDF
        return text

    def validate_urdu_content(self, content: str) -> Dict[str, Any]:
        """Validate Urdu content for quality and consistency"""
        validation_results = {
            'is_urdu': self.is_urdu_text(content),
            'character_analysis': self._analyze_characters(content),
            'number_format_consistency': self._check_number_formats(content),
            'punctuation_consistency': self._check_punctuation(content),
            'encoding_issues': self._check_encoding_issues(content),
            'overall_quality_score': 0.0
        }

        # Calculate quality score
        score = 0.0
        if validation_results['is_urdu']:
            score += 0.3
        if validation_results['character_analysis']['proper_script']:
            score += 0.2
        if validation_results['number_format_consistency']['consistent']:
            score += 0.2
        if validation_results['punctuation_consistency']['consistent']:
            score += 0.2
        if not validation_results['encoding_issues']:
            score += 0.1

        validation_results['overall_quality_score'] = min(score, 1.0)

        return validation_results

    def _analyze_characters(self, text: str) -> Dict[str, Any]:
        """Analyze character composition of Urdu text"""
        char_analysis = {
            'total_chars': len(text),
            'urdu_chars': 0,
            'latin_chars': 0,
            'numbers': 0,
            'punctuation': 0,
            'whitespace': 0,
            'other': 0,
            'proper_script': False
        }

        for char in text:
            if char in self.urdu_characters:
                char_analysis['urdu_chars'] += 1
            elif char.isalpha() and ord(char) < 256:  # Latin characters
                char_analysis['latin_chars'] += 1
            elif char.isdigit():
                char_analysis['numbers'] += 1
            elif char in '.,!?;:':
                char_analysis['punctuation'] += 1
            elif char.isspace():
                char_analysis['whitespace'] += 1
            else:
                char_analysis['other'] += 1

        # Consider script proper if majority of text is in Urdu script
        char_analysis['proper_script'] = (char_analysis['urdu_chars'] /
                                        max(char_analysis['total_chars'], 1)) > 0.3

        return char_analysis

    def _check_number_formats(self, text: str) -> Dict[str, Any]:
        """Check consistency of number formats in text"""
        western_count = sum(1 for c in text if c.isdigit())
        urdu_count = sum(1 for c in text if c in self.urdu_numerals.values())

        return {
            'western_numerals': western_count,
            'urdu_numerals': urdu_count,
            'consistent': (western_count == 0) or (urdu_count == 0),  # All one type
            'mixed_usage': min(western_count, urdu_count) > 0  # Both types present
        }

    def _check_punctuation(self, text: str) -> Dict[str, Any]:
        """Check consistency of punctuation marks"""
        urdu_punct_count = sum(1 for c in text if c in self.right_to_left_punctuation)
        latin_punct_count = sum(1 for c in text if c in '.,!?;:')

        return {
            'urdu_punctuation': urdu_punct_count,
            'latin_punctuation': latin_punct_count,
            'consistent': (urdu_punct_count == 0) or (latin_punct_count == 0),
            'mixed_usage': min(urdu_punct_count, latin_punct_count) > 0
        }

    def _check_encoding_issues(self, text: str) -> List[str]:
        """Check for common encoding issues in Urdu text"""
        issues = []

        # Check for common problematic character sequences
        problematic_sequences = [
            '\u064E\u064F',  # Fatha followed by Damma (often incorrect)
            '\u064F\u064E',  # Damma followed by Fatha (often incorrect)
        ]

        for seq in problematic_sequences:
            if seq in text:
                issues.append(f"Potentially incorrect character sequence: {repr(seq)}")

        # Check for unusual Unicode characters
        for i, char in enumerate(text):
            if 0x6FF < ord(char) < 0x750:  # Between Arabic and Syriac blocks
                issues.append(f"Unusual Unicode character at position {i}: {repr(char)}")

        return issues


class TechnicalTerminologyManager:
    """Manages technical terminology across languages"""

    def __init__(self):
        self.terminology_databases = {
            'robotics': {
                'en': {
                    'actuator': 'A component of a machine that causes it to move',
                    'end-effector': 'The device at the end of a robotic arm designed to interact with the environment',
                    'kinematics': 'The study of motion without considering the forces that cause it',
                    'dynamics': 'The study of motion with consideration of the forces that cause it',
                    'trajectory': 'The path that a moving object follows through space as a function of time',
                    'manipulator': 'A mechanical device designed to manipulate objects',
                    'locomotion': 'The ability to move from one place to another',
                    'perception': 'The process of acquiring sensory information from the environment',
                    'navigation': 'The process of planning and following paths in space',
                    'control': 'The process of commanding and regulating system behavior'
                },
                'ur': {
                    'actuator': 'عمل کنندہ',  # Actuator
                    'end-effector': 'آخری ایفکٹر',  # End-effector
                    'kinematics': 'کنیمیٹکس',  # Kinematics
                    'dynamics': 'ڈائنامکس',  # Dynamics
                    'trajectory': 'پیمائش',  # Trajectory
                    'manipulator': 'مینوولیٹر',  # Manipulator
                    'locomotion': 'حرکت',  # Locomotion
                    'perception': 'ادراک',  # Perception
                    'navigation': 'راہ نما',  # Navigation
                    'control': 'کنٹرول'  # Control
                }
            },
            'ai_ml': {
                'en': {
                    'neural_network': 'A computing system inspired by the biological neural networks',
                    'deep_learning': 'Part of machine learning based on artificial neural networks',
                    'reinforcement_learning': 'An area of machine learning concerned with how agents take actions',
                    'computer_vision': 'A field of artificial intelligence that trains computers to interpret and understand visual world',
                    'natural_language_processing': 'A subfield of linguistics, computer science, and artificial intelligence',
                    'algorithm': 'A step-by-step procedure for calculations',
                    'dataset': 'A collection of data used for analysis and training',
                    'model': 'A mathematical representation of a real-world process',
                    'training': 'The process of teaching a model using data',
                    'inference': 'The process of using a trained model to make predictions'
                },
                'ur': {
                    'neural_network': 'نیورل نیٹ ورک',  # Neural Network
                    'deep_learning': 'گہری سیکھ',  # Deep Learning (literal translation)
                    'reinforcement_learning': 'تقویت بخش سیکھ',  # Reinforcement Learning
                    'computer_vision': 'کمپیوٹر وژن',  # Computer Vision
                    'natural_language_processing': 'قدرتی زبان کی پروسیسنگ',  # Natural Language Processing
                    'algorithm': 'الگورتھم',  # Algorithm
                    'dataset': 'ڈیٹا سیٹ',  # Dataset
                    'model': 'ماڈل',  # Model
                    'training': 'تربیت',  # Training
                    'inference': 'استدلال'  # Inference
                }
            },
            'physics': {
                'en': {
                    'force': 'Any interaction that, when unopposed, will change the motion of an object',
                    'torque': 'The rotational equivalent of linear force',
                    'momentum': 'The product of the mass and velocity of an object',
                    'energy': 'The quantitative property that must be transferred to an object',
                    'velocity': 'The rate of change of displacement with respect to time',
                    'acceleration': 'The rate of change of velocity with respect to time',
                    'gravity': 'The force that attracts a body toward the center of the earth',
                    'friction': 'The force resisting the relative motion of solid surfaces',
                    'inertia': 'A property of matter by which it continues in its existing state',
                    'equilibrium': 'The state of a body at rest or in uniform motion'
                },
                'ur': {
                    'force': 'قوت',  # Force
                    'torque': 'ٹارک',  # Torque
                    'momentum': 'تھرتھرا',  # Momentum
                    'energy': 'توانائی',  # Energy
                    'velocity': 'رفعت',  # Velocity
                    'acceleration': 'تیزی',  # Acceleration
                    'gravity': 'کشش',  # Gravity
                    'friction': 'رگڑ',  # Friction
                    'inertia': 'لیت_lazy',  # Inertia (closest term)
                    'equilibrium': 'مساوات'  # Equilibrium
                }
            }
        }

    def get_domain_terms(self, source_lang: LanguageCode, target_lang: LanguageCode,
                         domain: str = 'robotics') -> Dict[str, str]:
        """Get all terms for a specific domain and language pair"""
        if domain not in self.terminology_databases:
            raise ValueError(f"Domain {domain} not supported")

        if source_lang.value not in self.terminology_databases[domain]:
            raise ValueError(f"Source language {source_lang} not supported for domain {domain}")

        if target_lang.value not in self.terminology_databases[domain]:
            raise ValueError(f"Target language {target_lang} not supported for domain {domain}")

        source_dict = self.terminology_databases[domain][source_lang.value]
        target_dict = self.terminology_databases[domain][target_lang.value]

        term_mapping = {}
        for term, _ in source_dict.items():
            if term in target_dict:
                term_mapping[term] = target_dict[term]

        return term_mapping

    def get_translation(self, term: str, source_lang: LanguageCode,
                       target_lang: LanguageCode, domain: str = 'robotics') -> str:
        """Get translation for a technical term"""
        if domain not in self.terminology_databases:
            raise ValueError(f"Domain {domain} not supported")

        if source_lang.value not in self.terminology_databases[domain]:
            raise ValueError(f"Source language {source_lang} not supported for domain {domain}")

        if target_lang.value not in self.terminology_databases[domain]:
            raise ValueError(f"Target language {target_lang} not supported for domain {domain}")

        # Get the dictionary for the source language
        source_dict = self.terminology_databases[domain][source_lang.value]

        # Find the term in the source dictionary
        for eng_term, eng_desc in source_dict.items():
            if eng_term == term or eng_desc == term:
                # Get the corresponding term in the target language
                target_dict = self.terminology_databases[domain][target_lang.value]

                # Find the corresponding term
                if term in target_dict:
                    return target_dict[term]
                else:
                    # If exact term not found, return the English term
                    return term

        # If term not found in source language, return original
        return term

    def validate_translation_consistency(self, content: str, source_lang: LanguageCode,
                                       target_lang: LanguageCode) -> List[str]:
        """Validate that technical terms are translated consistently"""
        issues = []

        # Extract technical terms from content
        terms_in_content = self._extract_technical_terms(content)

        # Check each term for consistency
        for term in terms_in_content:
            translation = self.get_translation(term, source_lang, target_lang)

            # Check if the translation is appropriate
            if translation == term and source_lang != target_lang:
                # Term wasn't translated - might be an issue
                issues.append(f"Term '{term}' was not translated from {source_lang.value} to {target_lang.value}")

        return issues

    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content"""
        # This is a simplified implementation
        # In practice, use NLP techniques for better term extraction
        all_terms = set()

        for domain, dicts in self.terminology_databases.items():
            for term in dicts.get('en', {}).keys():
                if term in content.lower():
                    all_terms.add(term)

        return list(all_terms)


class CulturalAdaptationProcessor:
    """Adapts content for cultural context"""

    def __init__(self):
        self.cultural_contexts = {
            'ur': {
                'examples': {
                    'robotics': [
                        'خودکار نظام برائے چاول کی کٹائی پاکستان میں',
                        'روبوٹک حل برائے لاہور میں ٹیکسٹائل کی پیداوار',
                        'AI طاقت ور نظم ال drip کے نظام برائے پنجاب کے کاشت کار'
                    ],
                    'education': [
                        'روبوٹکس کے ذریعے STEM سیکھنا پاکستانی اسکولوں میں',
                        'AI کی اطلاقات اردو زبان کی پروسیسنگ میں',
                        'روبوٹکس مقابلہ جات پاکستانی یونیورسٹیوں میں'
                    ]
                },
                'measurement_units': 'metric_system',  # Use metric system
                'calendar_system': 'gregorian',  # Gregorian calendar
                'writing_direction': 'rtl',  # Right-to-left for Arabic script
                'formality_level': 'high',  # High formality in academic settings
                'numeric_format': 'western_arabic',  # Use Western Arabic numerals
                'currency_format': 'pkr'  # Pakistani Rupee
            }
        }

    def adapt_content_for_culture(self, content: str, target_language: LanguageCode) -> str:
        """Adapt content for cultural context"""
        if target_language.value not in self.cultural_contexts:
            # Return original content if no cultural context available
            return content

        adapted_content = content

        # Adapt examples based on cultural context
        adapted_content = self._adapt_examples(adapted_content, target_language.value)

        # Adapt measurement units
        adapted_content = self._adapt_measurement_units(adapted_content, target_language.value)

        # Adapt formality level
        adapted_content = self._adapt_formality(adapted_content, target_language.value)

        return adapted_content

    def _adapt_examples(self, content: str, target_language: str) -> str:
        """Replace examples with culturally relevant ones"""
        if target_language == 'ur':
            # Replace generic examples with Pakistan/Urdu-specific examples
            replacements = {
                'American factory automation': 'Pakistani textile factory automation',
                'Silicon Valley robotics company': 'Pakistani robotics startup',
                'California university': 'Pakistani university',
                'Boston hospital': 'Karachi hospital',
                'Detroit manufacturing': 'Faisalabad manufacturing'
            }

            adapted_content = content
            for old_example, new_example in replacements.items():
                adapted_content = adapted_content.replace(old_example, new_example)

            return adapted_content

        return content

    def _adapt_measurement_units(self, content: str, target_language: str) -> str:
        """Convert measurement units to appropriate system"""
        if target_language == 'ur':
            # In Pakistan, metric system is standard, so no conversion needed
            # But we could add conversions if needed
            return content

        return content

    def _adapt_formality(self, content: str, target_language: str) -> str:
        """Adjust formality level for academic context"""
        if target_language == 'ur':
            # Urdu academic content typically uses more formal language
            formal_replacements = {
                'you': 'آپ',  # Formal 'you' in Urdu context
                'informal greeting': 'formal greeting',
                'casual tone': 'academic tone'
            }

            adapted_content = content
            for informal, formal in formal_replacements.items():
                adapted_content = adapted_content.replace(informal, formal)

            return adapted_content

        return content


class TranslationEngine:
    """Main translation engine that orchestrates multilingual processing"""

    def __init__(self):
        self.urdu_processor = UrduTextProcessor()
        self.terminology_manager = TechnicalTerminologyManager()
        self.cultural_processor = CulturalAdaptationProcessor()
        self.is_initialized = False

    async def initialize(self):
        """Initialize translation engine"""
        logger.info("Initializing Translation Engine")
        self.is_initialized = True
        logger.info("Translation Engine initialized successfully")

    async def translate_text(self, text: str, source_lang: LanguageCode,
                           target_lang: LanguageCode, domain: str = 'robotics') -> TranslationResult:
        """Translate text from source to target language"""
        if not self.is_initialized:
            raise RuntimeError("Translation engine not initialized")

        start_time = time.time()

        try:
            # For now, implement a mock translation
            # In a real implementation, this would use translation models or services

            if source_lang == LanguageCode.ENGLISH and target_lang == LanguageCode.URDU:
                translated_text = self._translate_en_to_ur(text, domain)
            elif source_lang == LanguageCode.URDU and target_lang == LanguageCode.ENGLISH:
                translated_text = self._translate_ur_to_en(text, domain)
            else:
                # For other language pairs, return original text as placeholder
                translated_text = text

            # Apply cultural adaptation
            culturally_adapted = self.cultural_processor.adapt_content_for_culture(
                translated_text, target_lang
            )

            # Apply Urdu-specific processing if target is Urdu
            if target_lang == LanguageCode.URDU:
                culturally_adapted = self.urdu_processor.handle_bidirectional_text(culturally_adapted)

            execution_time = time.time() - start_time

            # Calculate quality score (mock implementation)
            quality_score = self._calculate_translation_quality(text, culturally_adapted)

            return TranslationResult(
                original_text=text,
                translated_text=culturally_adapted,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.85,  # Mock confidence
                translation_time=execution_time,
                quality_score=quality_score
            )

        except Exception as e:
            execution_time = time.time() - start_time
            raise RuntimeError(f"Translation failed: {str(e)}")

    def _translate_en_to_ur(self, text: str, domain: str) -> str:
        """Translate English to Urdu using terminology database"""
        # In a real implementation, this would use proper translation models
        # For now, implement a basic term-by-term translation

        # Get domain-specific terminology
        terms = self.terminology_manager.get_domain_terms(
            LanguageCode.ENGLISH, LanguageCode.URDU, domain
        )

        # Replace terms in the text
        translated_text = text
        for eng_term, urdu_term in terms.items():
            # Case-insensitive replacement
            translated_text = re.sub(
                r'\b' + re.escape(eng_term) + r'\b',
                urdu_term,
                translated_text,
                flags=re.IGNORECASE
            )

        return translated_text

    def _translate_ur_to_en(self, text: str, domain: str) -> str:
        """Translate Urdu to English using terminology database"""
        # Get domain-specific terminology
        terms = self.terminology_manager.get_domain_terms(
            LanguageCode.URDU, LanguageCode.ENGLISH, domain
        )

        # Replace terms in the text
        translated_text = text
        for urdu_term, eng_term in terms.items():
            # Exact replacement (case-sensitive for Urdu)
            translated_text = translated_text.replace(urdu_term, eng_term)

        return translated_text

    def _calculate_translation_quality(self, original: str, translated: str) -> float:
        """Calculate translation quality score"""
        # Mock quality calculation
        # In real implementation, use BLEU, METEOR, or other quality metrics
        if len(original) == 0:
            return 0.0

        # Simple length-based similarity
        length_ratio = min(len(translated), len(original)) / max(len(translated), len(original))

        # Check for Urdu characters if translating to Urdu
        if any(c in self.urdu_processor.urdu_characters for c in translated):
            length_ratio *= 1.2  # Boost for using proper script

        return min(length_ratio, 1.0)

    async def translate_document(self, document: str, source_lang: LanguageCode,
                               target_lang: LanguageCode, domain: str = 'robotics') -> TranslationResult:
        """Translate entire document with context preservation"""
        if not self.is_initialized:
            raise RuntimeError("Translation engine not initialized")

        start_time = time.time()

        # Split document into segments to preserve context
        segments = self._split_document_into_segments(document)

        translated_segments = []
        for segment in segments:
            translation_result = await self.translate_text(
                segment, source_lang, target_lang, domain
            )
            translated_segments.append(translation_result.translated_text)

        final_translation = ' '.join(translated_segments)

        execution_time = time.time() - start_time

        # Calculate overall quality
        overall_quality = sum(
            self._calculate_translation_quality(seg, trans_seg)
            for seg, trans_seg in zip(segments, translated_segments)
        ) / len(segments) if segments else 0.0

        return TranslationResult(
            original_text=document,
            translated_text=final_translation,
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.85,
            translation_time=execution_time,
            quality_score=overall_quality
        )

    def _split_document_into_segments(self, document: str) -> List[str]:
        """Split document into segments for translation"""
        # Simple segmentation by sentences
        # In practice, use more sophisticated segmentation that considers context
        import re
        sentences = re.split(r'[.!?]+', document)
        return [s.strip() for s in sentences if s.strip()]


class LocalizationEngine:
    """Handles localization beyond translation (cultural adaptation, units, etc.)"""

    def __init__(self):
        self.translation_engine = None  # Will be set during initialization
        self.urdu_processor = UrduTextProcessor()

    async def initialize(self, translation_engine: TranslationEngine):
        """Initialize localization engine with translation engine"""
        self.translation_engine = translation_engine
        logger.info("Localization Engine initialized successfully")

    async def localize_content(self, content: str, target_language: LanguageCode) -> LocalizationResult:
        """Localize content for target language and culture"""
        if not self.translation_engine:
            raise RuntimeError("Localization engine not initialized")

        start_time = time.time()

        # For now, implement basic localization
        localized_content = content

        # Apply cultural adaptations
        cultural_adaptations = []
        if target_language == LanguageCode.URDU:
            # Adapt for Pakistani/Urdu cultural context
            localized_content = self._adapt_for_urdu_culture(localized_content)
            cultural_adaptations.append("Cultural examples adapted for Pakistani context")
            cultural_adaptations.append("Formality level adjusted for academic Urdu")

        # Apply measurement unit conversions
        measurement_units = "metric"  # Default
        if target_language in [LanguageCode.URDU]:
            localized_content = self._convert_to_metric_units(localized_content)
            measurement_units = "metric"

        # Apply currency formatting
        currency_format = "PKR" if target_language == LanguageCode.URDU else "USD"

        # Apply date formatting
        date_format = "DD/MM/YYYY" if target_language == LanguageCode.URDU else "MM/DD/YYYY"

        execution_time = time.time() - start_time

        return LocalizationResult(
            localized_content=localized_content,
            cultural_adaptations=cultural_adaptations,
            measurement_units=measurement_units,
            currency_format=currency_format,
            date_format=date_format
        )

    def _adapt_for_urdu_culture(self, content: str) -> str:
        """Adapt content for Urdu/Pakistani cultural context"""
        # Replace examples with Pakistani context
        replacements = {
            "Silicon Valley": "Pakistani tech hubs",
            "American university": "Pakistani university",
            "Boston hospital": "Karachi hospital",
            "Detroit manufacturing": "Faisalabad manufacturing",
            "California agriculture": "Punjab agriculture"
        }

        localized_content = content
        for old, new in replacements.items():
            localized_content = localized_content.replace(old, new)

        return localized_content

    def _convert_to_metric_units(self, content: str) -> str:
        """Convert imperial units to metric units"""
        # This is a simplified implementation
        # In practice, use more sophisticated unit conversion
        conversions = {
            r'(\d+(?:\.\d+)?)\s*ft': lambda m: f"{float(m.group(1)) * 0.3048:.2f} m",
            r'(\d+(?:\.\d+)?)\s*miles': lambda m: f"{float(m.group(1)) * 1.60934:.2f} km",
            r'(\d+(?:\.\d+)?)\s*lbs': lambda m: f"{float(m.group(1)) * 0.453592:.2f} kg",
            r'(\d+(?:\.\d+)?)\s*inches': lambda m: f"{float(m.group(1)) * 2.54:.2f} cm"
        }

        localized_content = content
        for pattern, conversion_func in conversions.items():
            localized_content = re.sub(pattern, conversion_func, localized_content, flags=re.IGNORECASE)

        return localized_content


class MultilingualContentManager:
    """Manages multilingual content throughout the system"""

    def __init__(self):
        self.translation_engine = TranslationEngine()
        self.localization_engine = LocalizationEngine()
        self.content_store = {}  # In practice, use a proper database
        self.is_initialized = False

    async def initialize(self):
        """Initialize multilingual content manager"""
        logger.info("Initializing Multilingual Content Manager")

        await self.translation_engine.initialize()
        await self.localization_engine.initialize(self.translation_engine)

        self.is_initialized = True
        logger.info("Multilingual Content Manager initialized successfully")

    async def add_content(self, content_id: str, content: str, language: LanguageCode):
        """Add content in a specific language"""
        if not self.is_initialized:
            raise RuntimeError("Multilingual content manager not initialized")

        if content_id not in self.content_store:
            self.content_store[content_id] = {}

        self.content_store[content_id][language.value] = content

    async def get_content(self, content_id: str, target_language: LanguageCode,
                         source_language: LanguageCode = LanguageCode.ENGLISH) -> str:
        """Get content in target language, translating if necessary"""
        if not self.is_initialized:
            raise RuntimeError("Multilingual content manager not initialized")

        # Check if content exists in target language
        if content_id in self.content_store:
            if target_language.value in self.content_store[content_id]:
                return self.content_store[content_id][target_language.value]

        # If content doesn't exist in target language, translate from source language
        if content_id in self.content_store:
            if source_language.value in self.content_store[content_id]:
                source_content = self.content_store[content_id][source_language.value]

                # Translate to target language
                translation_result = await self.translation_engine.translate_text(
                    source_content, source_language, target_language
                )

                # Store translated content for future use
                await self.add_content(content_id, translation_result.translated_text, target_language)

                return translation_result.translated_text

        # If content doesn't exist at all, return error
        raise ValueError(f"Content with ID {content_id} not found")

    async def get_available_languages(self, content_id: str) -> List[LanguageCode]:
        """Get list of available languages for specific content"""
        if content_id in self.content_store:
            return [LanguageCode(code) for code in self.content_store[content_id].keys()]
        return []

    async def translate_content(self, content_id: str, target_languages: List[LanguageCode],
                              source_language: LanguageCode = LanguageCode.ENGLISH):
        """Translate content to multiple target languages"""
        if not self.is_initialized:
            raise RuntimeError("Multilingual content manager not initialized")

        if content_id not in self.content_store:
            raise ValueError(f"Content with ID {content_id} not found")

        if source_language.value not in self.content_store[content_id]:
            raise ValueError(f"Source language {source_language.value} not available for content {content_id}")

        source_content = self.content_store[content_id][source_language.value]

        for target_lang in target_languages:
            if target_lang.value not in self.content_store[content_id]:
                # Translate and store
                translation_result = await self.translation_engine.translate_text(
                    source_content, source_language, target_lang
                )
                self.content_store[content_id][target_lang.value] = translation_result.translated_text

    async def validate_content_translation(self, content_id: str, target_language: LanguageCode) -> Dict[str, Any]:
        """Validate that content translation is accurate and culturally appropriate"""
        if not self.is_initialized:
            raise RuntimeError("Multilingual content manager not initialized")

        if content_id not in self.content_store:
            raise ValueError(f"Content with ID {content_id} not found")

        if target_language.value not in self.content_store[content_id]:
            raise ValueError(f"Content {content_id} not available in {target_language.value}")

        content = self.content_store[content_id][target_language.value]

        validation_results = {}

        if target_language == LanguageCode.URDU:
            # Validate Urdu content
            validation_results['urdu_validation'] = self.urdu_processor.validate_urdu_content(content)

        # Validate technical terminology consistency
        validation_results['terminology_consistency'] = (
            self.terminology_manager.validate_translation_consistency(
                content, LanguageCode.ENGLISH, target_language
            )
        )

        return validation_results

    def get_health_status(self) -> str:
        """Get health status of multilingual system"""
        if not self.is_initialized:
            return "not_initialized"
        return "healthy"

    async def shutdown(self):
        """Shutdown multilingual content manager"""
        logger.info("Shutting down Multilingual Content Manager")
        self.is_initialized = False
        logger.info("Multilingual Content Manager shutdown complete")