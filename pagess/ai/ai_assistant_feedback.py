# pagess/ai_assistant_feedback.py
"""
Smart Feedback System for Educational AI
Tracks utility + behavior + optional validation
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Optional
import json
from pathlib import Path

class FeedbackTracker:
    """Track AI response quality without requiring expert knowledge"""
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "ai_feedback.json"
        
        # Load existing feedback
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        """Load feedback history"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'responses': [], 'stats': {}}
        return {'responses': [], 'stats': {}}
    
    def _save_feedback(self):
        """Save feedback to disk"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
    
    def render_feedback_ui(self, message_id: str, response_text: str, query: str):
        """
        Render smart feedback UI
        
        Combines:
        - Quick utility rating (no expertise needed)
        - Behavioral tracking (automatic)
        - Optional detailed feedback
        """
        
        # Create unique key for this message
        feedback_key = f"feedback_{message_id}"
        
        # Check if already rated
        if st.session_state.get(f"{feedback_key}_rated", False):
            st.success("✅ Merci pour votre feedback !")
            return
        
        # Main feedback UI
        st.markdown("---")
        
        # 1. QUICK UTILITY (main metric)
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            st.markdown("**Cette réponse vous a été utile ?**")
        
        with col2:
            if st.button("👍 Oui", key=f"{feedback_key}_yes", use_container_width=True):
                self._record_feedback(message_id, query, response_text, {
                    'useful': True,
                    'rating': 'positive'
                })
                st.session_state[f"{feedback_key}_rated"] = True
                st.rerun()
        
        with col3:
            if st.button("👎 Non", key=f"{feedback_key}_no", use_container_width=True):
                # Show detailed feedback form
                st.session_state[f"{feedback_key}_show_details"] = True
                st.rerun()
        
        with col4:
            if st.button("⏭️ Passer", key=f"{feedback_key}_skip", use_container_width=True):
                st.session_state[f"{feedback_key}_rated"] = True
                st.rerun()
        
        # 2. DETAILED FEEDBACK (if negative or requested)
        if st.session_state.get(f"{feedback_key}_show_details", False):
            self._render_detailed_feedback(message_id, query, response_text, feedback_key)
    
    def _render_detailed_feedback(self, message_id: str, query: str, 
                                   response_text: str, feedback_key: str):
        """Detailed feedback form (only shown on negative feedback)"""
        
        with st.expander("💬 Dites-nous plus", expanded=True):
            st.markdown("**Qu'est-ce qui n'allait pas ?**")
            
            issues = st.multiselect(
                "Sélectionnez un ou plusieurs :",
                [
                    "❌ Pas compris l'explication",
                    "📊 Pas assez d'exemples concrets",
                    "🎯 Hors sujet",
                    "📚 Trop technique",
                    "😴 Trop simple",
                    "🔗 Manque de sources",
                    "❓ N'a pas répondu à ma question",
                    "🐛 Erreur factuelle (si vous êtes sûr)"
                ],
                key=f"{feedback_key}_issues"
            )
            
            comment = st.text_area(
                "Commentaire (optionnel) :",
                placeholder="Qu'auriez-vous aimé voir dans la réponse ?",
                key=f"{feedback_key}_comment"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("✅ Envoyer", key=f"{feedback_key}_submit", use_container_width=True):
                    self._record_feedback(message_id, query, response_text, {
                        'useful': False,
                        'rating': 'negative',
                        'issues': issues,
                        'comment': comment
                    })
                    st.session_state[f"{feedback_key}_rated"] = True
                    st.session_state[f"{feedback_key}_show_details"] = False
                    st.success("🙏 Merci ! Cela nous aide à nous améliorer.")
                    st.rerun()
            
            with col2:
                if st.button("Annuler", key=f"{feedback_key}_cancel", use_container_width=True):
                    st.session_state[f"{feedback_key}_show_details"] = False
                    st.rerun()
    
    def _record_feedback(self, message_id: str, query: str, 
                         response_text: str, feedback: Dict):
        """Record feedback in database"""
        
        feedback_entry = {
            'message_id': message_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_preview': response_text[:200] + '...' if len(response_text) > 200 else response_text,
            'user_id': st.session_state.get('user_id', 'anonymous'),
            **feedback
        }
        
        self.feedback_data['responses'].append(feedback_entry)
        
        # Update stats
        rating = feedback.get('rating', 'neutral')
        stats = self.feedback_data.get('stats', {})
        stats[rating] = stats.get(rating, 0) + 1
        self.feedback_data['stats'] = stats
        
        self._save_feedback()
    
    def track_behavior(self, message_id: str, action: str, metadata: Optional[Dict] = None):
        """
        Track user behavior automatically (no UI)
        
        Actions:
        - 'copied_text': User copied response
        - 'follow_up': Asked follow-up question
        - 'action_taken': Created portfolio, analyzed, etc.
        - 'abandoned': Left conversation
        """
        
        behavior_entry = {
            'message_id': message_id,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user_id': st.session_state.get('user_id', 'anonymous'),
            'metadata': metadata or {}
        }
        
        if 'behaviors' not in self.feedback_data:
            self.feedback_data['behaviors'] = []
        
        self.feedback_data['behaviors'].append(behavior_entry)
        self._save_feedback()
    
    def get_response_quality_score(self, message_id: str) -> Optional[float]:
        """
        Calculate quality score for a response
        
        Combines:
        - Explicit feedback (thumbs up/down)
        - Behavioral signals
        
        Returns: 0.0 (bad) to 1.0 (excellent)
        """
        
        # Find explicit feedback
        explicit_score = None
        for entry in self.feedback_data.get('responses', []):
            if entry['message_id'] == message_id:
                if entry['rating'] == 'positive':
                    explicit_score = 1.0
                elif entry['rating'] == 'negative':
                    explicit_score = 0.0
                break
        
        # Find behavioral signals
        positive_behaviors = 0
        negative_behaviors = 0
        
        for behavior in self.feedback_data.get('behaviors', []):
            if behavior['message_id'] == message_id:
                action = behavior['action']
                if action in ['copied_text', 'follow_up', 'action_taken']:
                    positive_behaviors += 1
                elif action in ['abandoned', 'reformulated_immediately']:
                    negative_behaviors += 1
        
        # Combine signals
        if explicit_score is not None:
            # Explicit feedback is stronger
            behavioral_score = (positive_behaviors - negative_behaviors) * 0.1
            return max(0.0, min(1.0, explicit_score + behavioral_score))
        
        elif positive_behaviors > 0 or negative_behaviors > 0:
            # Only behavioral
            return max(0.0, min(1.0, 0.5 + (positive_behaviors - negative_behaviors) * 0.2))
        
        return None  # No data
    
    def get_stats(self) -> Dict:
        """Get feedback statistics"""
        stats = self.feedback_data.get('stats', {})
        total = sum(stats.values())
        
        if total == 0:
            return {
                'total_feedback': 0,
                'positive_rate': 0.0,
                'negative_rate': 0.0
            }
        
        return {
            'total_feedback': total,
            'positive_rate': stats.get('positive', 0) / total,
            'negative_rate': stats.get('negative', 0) / total,
            'positive': stats.get('positive', 0),
            'negative': stats.get('negative', 0)
        }
    
    def get_improvement_suggestions(self) -> list:
        """Analyze feedback to suggest improvements"""
        
        suggestions = []
        
        # Analyze negative feedback
        issue_counts = {}
        for entry in self.feedback_data.get('responses', []):
            if entry.get('rating') == 'negative' and entry.get('issues'):
                for issue in entry['issues']:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        for issue, count in sorted_issues[:3]:  # Top 3 issues
            suggestions.append({
                'issue': issue,
                'count': count,
                'suggestion': self._get_suggestion_for_issue(issue)
            })
        
        return suggestions
    
    def _get_suggestion_for_issue(self, issue: str) -> str:
        """Map issue to improvement suggestion"""
        
        suggestions_map = {
            "❌ Pas compris l'explication": "Simplifier le langage, ajouter plus d'analogies",
            "📊 Pas assez d'exemples concrets": "Ajouter 2-3 exemples chiffrés systématiquement",
            "🎯 Hors sujet": "Améliorer la détection d'intention dans les requêtes",
            "📚 Trop technique": "Créer une version 'explain like I'm 5' par défaut",
            "😴 Trop simple": "Proposer option 'mode avancé' pour les experts",
            "🔗 Manque de sources": "Toujours inclure 2-3 sources externes",
            "❓ N'a pas répondu à ma question": "Vérifier la pertinence de la réponse",
            "🐛 Erreur factuelle": "URGENT: Vérifier et corriger les erreurs factuelles"
        }
        
        return suggestions_map.get(issue, "Analyser ce cas spécifique")


# =============================================================================
# Integration with AI Assistant
# =============================================================================

def add_feedback_to_chat_message(message_id: str, response_text: str, query: str):
    """
    Add feedback UI after AI response
    Call this after displaying AI message
    """
    
    # Initialize tracker
    if 'feedback_tracker' not in st.session_state:
        st.session_state.feedback_tracker = FeedbackTracker()
    
    tracker = st.session_state.feedback_tracker
    
    # Render feedback UI
    tracker.render_feedback_ui(message_id, response_text, query)


def track_user_action(message_id: str, action: str, metadata: Optional[Dict] = None):
    """
    Track user behavior automatically
    Call this when user performs actions
    """
    
    if 'feedback_tracker' not in st.session_state:
        st.session_state.feedback_tracker = FeedbackTracker()
    
    tracker = st.session_state.feedback_tracker
    tracker.track_behavior(message_id, action, metadata)


def show_feedback_dashboard():
    """Show feedback dashboard in sidebar (admin)"""
    
    if 'feedback_tracker' not in st.session_state:
        st.session_state.feedback_tracker = FeedbackTracker()
    
    tracker = st.session_state.feedback_tracker
    stats = tracker.get_stats()
    
    with st.sidebar:
        with st.expander("📊 AI Feedback Stats"):
            st.metric("Total Feedback", stats['total_feedback'])
            
            if stats['total_feedback'] > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("👍 Positif", f"{stats['positive_rate']*100:.0f}%")
                with col2:
                    st.metric("👎 Négatif", f"{stats['negative_rate']*100:.0f}%")
                
                # Improvement suggestions
                suggestions = tracker.get_improvement_suggestions()
                if suggestions:
                    st.markdown("**🔧 À améliorer :**")
                    for sug in suggestions[:2]:
                        st.info(f"{sug['issue']} ({sug['count']}x)\n→ {sug['suggestion']}")


# =============================================================================
# Usage Example
# =============================================================================

"""
# In ai_assistant.py, after displaying AI response:

# 1. Generate unique message ID
import hashlib
message_id = hashlib.md5(f"{user_query}{timestamp}".encode()).hexdigest()

# 2. Display AI response
st.markdown(ai_response)

# 3. Add feedback UI
add_feedback_to_chat_message(message_id, ai_response, user_query)

# 4. Track behaviors (automatic)
# When user copies text:
track_user_action(message_id, 'copied_text')

# When user asks follow-up:
track_user_action(message_id, 'follow_up', {'follow_up_query': next_query})

# When user takes action:
track_user_action(message_id, 'action_taken', {'action': 'created_portfolio'})

# 5. Show dashboard (sidebar)
show_feedback_dashboard()
"""
