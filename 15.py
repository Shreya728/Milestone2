import whisper
import moviepy.editor as mp
from transformers import pipeline
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import logging
from typing import Optional, List, Dict, Tuple, Union
import re
from collections import defaultdict
import spacy
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class MeetingAnalyzer:
    def __init__(self):
        """Initialize the analysis components."""
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # Load spaCy model for NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
        
        # Keywords for meeting genre classification
        self.genre_keywords = {
            "Annual General Meeting": ["agm", "annual general meeting", "shareholders", "dividend", "financial year"],
            "Board Meeting": ["board", "directors", "governance", "resolution"],
            "Team Meeting": ["team", "project", "updates", "progress"],
            "Sales Meeting": ["sales", "revenue", "customers", "targets"],
            "Strategy Meeting": ["strategy", "planning", "roadmap", "objectives"]
        }

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the overall sentiment of the meeting."""
        try:
            # Split text into smaller chunks for analysis
            chunks = text.split('. ')
            sentiments = []
            
            for chunk in chunks:
                if len(chunk.strip()) > 0:
                    result = self.sentiment_analyzer(chunk[:512])[0]
                    # Convert 1-5 scale to sentiment score
                    score = float(result['label'].split()[0]) / 5.0
                    sentiments.append(score)
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            return {
                "score": avg_sentiment,
                "label": self._get_sentiment_label(avg_sentiment),
                "confidence": sum(1 for s in sentiments if abs(s - avg_sentiment) < 0.2) / len(sentiments)
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"score": 0, "label": "neutral", "confidence": 0}

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score >= 0.75: return "Very Positive"
        elif score >= 0.6: return "Positive"
        elif score >= 0.4: return "Neutral"
        elif score >= 0.25: return "Negative"
        else: return "Very Negative"

    def count_speakers(self, text: str) -> Dict[str, any]:
        """Identify and count unique speakers in the transcript."""
        try:
            doc = self.nlp(text)
            speaker_patterns = [
                r"(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?=\s*:)",
            ]
            
            speakers = set()
            for pattern in speaker_patterns:
                matches = re.finditer(pattern, text)
                speakers.update(match.group(0) for match in matches)
            
            return {
                "count": len(speakers),
                "speakers": list(speakers),
                "confidence": "high" if len(speakers) > 0 else "low"
            }
        except Exception as e:
            logger.error(f"Error in speaker detection: {e}")
            return {"count": 0, "speakers": [], "confidence": "low"}

    def determine_genre(self, text: str) -> Dict[str, any]:
        """Determine the type/genre of the meeting."""
        try:
            text_lower = text.lower()
            scores = defaultdict(int)
            total_keywords = 0
            
            # Count keyword matches for each genre
            for genre, keywords in self.genre_keywords.items():
                for keyword in keywords:
                    count = text_lower.count(keyword)
                    scores[genre] += count
                    total_keywords += count
            
            # Calculate probabilities and find the best match
            if total_keywords > 0:
                genre_probs = {
                    genre: count / total_keywords 
                    for genre, count in scores.items()
                }
                best_genre = max(genre_probs.items(), key=lambda x: x[1])
                
                return {
                    "genre": best_genre[0],
                    "confidence": best_genre[1],
                    "other_possibilities": sorted(
                        [(g, p) for g, p in genre_probs.items() if g != best_genre[0]],
                        key=lambda x: x[1],
                        reverse=True
                    )[:2]
                }
            return {"genre": "Unknown", "confidence": 0, "other_possibilities": []}
        except Exception as e:
            logger.error(f"Error in genre detection: {e}")
            return {"genre": "Unknown", "confidence": 0, "other_possibilities": []}

    def extract_action_items(self, text: str) -> List[Dict[str, str]]:
        """Extract action items and their assignments from the transcript."""
        try:
            doc = self.nlp(text)
            action_items = []
            
            # Keywords that might indicate action items
            action_keywords = ["will", "shall", "must", "need to", "should", "have to", "going to"]
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                # Check if sentence contains action keywords
                if any(keyword in sent_text for keyword in action_keywords):
                    # Extract the action and responsible person if mentioned
                    action = {
                        "action": sent.text.strip(),
                        "assigned_to": None,
                        "deadline": None,
                        "priority": self._determine_priority(sent_text)
                    }
                    
                    # Look for person names in the sentence
                    for ent in sent.ents:
                        if ent.label_ == "PERSON":
                            action["assigned_to"] = ent.text
                    
                    # Look for dates
                    for ent in sent.ents:
                        if ent.label_ == "DATE":
                            action["deadline"] = ent.text
                    
                    action_items.append(action)
            
            return action_items
        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return []

    def _determine_priority(self, text: str) -> str:
        """Determine priority of an action item based on language used."""
        high_priority = ["urgent", "immediate", "asap", "critical", "crucial"]
        medium_priority = ["important", "necessary", "should", "needed"]
        
        text_lower = text.lower()
        if any(word in text_lower for word in high_priority):
            return "High"
        elif any(word in text_lower for word in medium_priority):
            return "Medium"
        return "Normal"

class MeetingProcessor:
    def __init__(self, model_name: str = "base"):
        """Initialize the meeting processor with necessary models."""
        self.whisper_model = whisper.load_model(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize summarizer and analyzer
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
            max_length=1024
        )
        self.analyzer = MeetingAnalyzer()

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        try:
            # Create output path for audio
            audio_path = video_path.rsplit('.', 1)[0] + '.wav'
            
            # Load video and extract audio
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            video.close()
            
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        try:
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def generate_summary(self, text: str) -> str:
        """Generate a summary of the meeting transcript."""
        try:
            # Split text into chunks that fit within model's max input length
            max_chunk_size = 1024
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            # Generate summary for each chunk
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 0:
                    summary = self.summarizer(chunk, max_length=120, min_length=30)[0]['summary_text']
                    summaries.append(summary)
            
            # Combine summaries
            return " ".join(summaries)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary could not be generated."

    def send_email(self, sender: str, password: str, receivers: Union[str, List[str]], subject: str, body: str) -> None:
        """Send an email with the meeting analysis report."""
        try:
            # Setup email server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender, password)

            # Create email
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(receivers) if isinstance(receivers, list) else receivers
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server.send_message(msg)
            server.quit()
            logger.info("Email sent successfully.")
        except Exception as e:
            logger.error(f"Error sending email: {e}")

    def process_meeting(
        self,
        video_path: str,
        email_sender: str,
        email_password: str,
        email_receivers: Union[str, List[str]]
    ) -> None:
        """Process meeting video and send summary with analysis."""
        try:
            # Extract audio and transcribe
            logger.info("Extracting audio from video...")
            audio_path = self.extract_audio(video_path)

            logger.info("Transcribing audio...")
            transcript = self.transcribe_audio(audio_path)

            # Perform analysis
            logger.info("Analyzing meeting content...")
            summary = self.generate_summary(transcript)
            sentiment = self.analyzer.analyze_sentiment(transcript)
            speakers = self.analyzer.count_speakers(transcript)
            genre = self.analyzer.determine_genre(transcript)
            action_items = self.analyzer.extract_action_items(transcript)

            # Prepare email content with enhanced genre information
            subject = f"Meeting Summary and Analysis - {genre['genre']}"
            body = f"""
Meeting Analysis Report

1. Meeting Genre Analysis:
- Primary Genre: {genre['genre']} (Confidence: {genre['confidence']:.2%})
- Other Possible Genres:
{self._format_genre_possibilities(genre['other_possibilities'])}

2. Summary:
{summary}

3. Meeting Details:
- Number of Speakers: {speakers['count']}
- Overall Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})
- Sentiment Confidence: {sentiment['confidence']:.2%} level of agreement in sentiment

4. Action Items:
{"" if not action_items else ""}
{self._format_action_items(action_items)}

5. Speakers Identified:
{self._format_speakers(speakers['speakers'])}

6. Full Transcript:
{transcript}

This is an automated analysis generated by the Meeting Processor.
            """
            logger.info("Sending email...")
            self.send_email(
                email_sender,
                email_password,
                email_receivers,
                subject,
                body
            )
        except Exception as e:
            logger.error(f"Error processing meeting: {e}")
            raise
    def _format_action_items(self, action_items: List[Dict[str, str]]) -> str:
        """Format action items for email."""
        if not action_items:
            return "No specific action items identified." 
        formatted = []
        for i, item in enumerate(action_items, 1):
            formatted.append(f"""
Action Item #{i}:
- Task: {item['action']}
- Assigned to: {item['assigned_to'] or 'Not specified'}
- Deadline: {item['deadline'] or 'Not specified'}
- Priority: {item['priority']}
""")
        return "\n".join(formatted)
    def _format_speakers(self, speakers: List[str]) -> str:
        """Format speaker list for email."""
        if not speakers:
            return "No specific speakers identified."
        return "\n".join(f"- {speaker}" for speaker in speakers)
    def _format_genre_possibilities(self, possibilities: List[Tuple[str, float]]) -> str:
        """Format alternative genre possibilities for email."""
        if not possibilities:
            return "  No other likely genres identified."
        return "\n".join(f"  - {genre}: {confidence:.2%} confidence" 
                        for genre, confidence in possibilities)
def main():
    VIDEO_PATH = "/Users/shreyaaa/Annual General Meeting FY 2019-20 copy.mp4"
    EMAIL_SENDER = "shreyapateriya728@gmail.com"
    EMAIL_PASSWORD = "edip fxpo qguh rqjw"  
    EMAIL_RECEIVERS = [
        "pradeeppateria0606@gmail.com",
        "shreya.pateriya2022@vitstudent.ac.in",
        "mamtapatriya728@gmail.com"
    ]
    try:
        processor = MeetingProcessor()
        processor.process_meeting(
            VIDEO_PATH,
            EMAIL_SENDER,
            EMAIL_PASSWORD,
            EMAIL_RECEIVERS
        )
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
if __name__ == "__main__":
    main()