import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .translation import translation_service

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    user_id: int
    background: str  # Educational/professional background
    preferred_difficulty: str  # beginner, intermediate, advanced
    learning_style: str  # visual, auditory, kinesthetic, reading/writing
    interests: List[str]
    goals: List[str]


@dataclass
class PersonalizationResult:
    modified_content: str
    adaptation_reasoning: str
    difficulty_level: str


class PersonalizationService:
    """
    Service for personalizing content based on user profiles
    """

    def __init__(self):
        self.translation_service = translation_service

    async def personalize_content(
        self,
        original_content: str,
        user_profile: UserProfile,
        content_type: str = "chapter"
    ) -> PersonalizationResult:
        """
        Personalize content based on user profile
        """
        try:
            # Apply personalization based on user profile
            modified_content = await self._apply_personalization(
                original_content,
                user_profile,
                content_type
            )

            # Determine adaptation reasoning
            reasoning = await self._generate_adaptation_reasoning(user_profile)

            return PersonalizationResult(
                modified_content=modified_content,
                adaptation_reasoning=reasoning,
                difficulty_level=user_profile.preferred_difficulty
            )

        except Exception as e:
            logger.error(f"Error personalizing content: {str(e)}")
            # Return original content if personalization fails
            return PersonalizationResult(
                modified_content=original_content,
                adaptation_reasoning="Personalization failed, returning original content",
                difficulty_level="original"
            )

    async def _apply_personalization(
        self,
        content: str,
        user_profile: UserProfile,
        content_type: str
    ) -> str:
        """
        Apply personalization transformations to content
        """
        modified_content = content

        # Adjust difficulty based on user preference
        modified_content = await self._adjust_difficulty(
            modified_content,
            user_profile.preferred_difficulty
        )

        # Adapt based on learning style
        modified_content = await self._adapt_for_learning_style(
            modified_content,
            user_profile.learning_style
        )

        # Add relevant examples based on background
        modified_content = await self._add_relevant_examples(
            modified_content,
            user_profile.background
        )

        return modified_content

    async def _adjust_difficulty(self, content: str, difficulty: str) -> str:
        """
        Adjust content difficulty level
        """
        if difficulty == "beginner":
            # Add more explanations and simpler language
            return f"[BEGINNER LEVEL ADAPTATION]\n{content}"
        elif difficulty == "advanced":
            # Add more technical depth
            return f"[ADVANCED LEVEL ADAPTATION]\n{content}"
        else:
            # Intermediate level - keep as is but with minor adjustments
            return f"[INTERMEDIATE LEVEL ADAPTATION]\n{content}"

    async def _adapt_for_learning_style(self, content: str, learning_style: str) -> str:
        """
        Adapt content for different learning styles
        """
        if learning_style == "visual":
            # Add more diagrams, charts, and visual elements (in practice)
            return f"[VISUAL LEARNING ADAPTATION]\n{content}"
        elif learning_style == "auditory":
            # Add more narrative and storytelling elements
            return f"[AUDITORY LEARNING ADAPTATION]\n{content}"
        elif learning_style == "kinesthetic":
            # Add more hands-on and practical elements
            return f"[KINESTHETIC LEARNING ADAPTATION]\n{content}"
        else:  # reading/writing
            return f"[READING/WRITING LEARNING ADAPTATION]\n{content}"

    async def _add_relevant_examples(self, content: str, background: str) -> str:
        """
        Add relevant examples based on user background
        """
        if background:
            return f"[EXAMPLES RELEVANT TO {background.upper()} BACKGROUND]\n{content}"
        return content

    async def _generate_adaptation_reasoning(self, user_profile: UserProfile) -> str:
        """
        Generate explanation for why content was adapted
        """
        return (
            f"Content adapted for user with {user_profile.background} background, "
            f"preferred difficulty level '{user_profile.preferred_difficulty}', "
            f"and {user_profile.learning_style} learning style."
        )

    async def get_personalization_recommendations(
        self,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Get recommendations for personalizing the learning experience
        """
        return {
            "difficulty_suggestion": user_profile.preferred_difficulty,
            "learning_style_adaptations": [user_profile.learning_style],
            "recommended_topics": user_profile.interests,
            "goal_alignment": user_profile.goals,
            "background_connection": f"Content will be contextualized with examples from {user_profile.background}"
        }


# Singleton instance
personalization_service = PersonalizationService()