import json
import os

class MemoryManager:
    def __init__(self, short_path="short_memory.json",
                 long_path="long_memory.json",
                 dynamic_path="dynamic_memory.json"):
        
        self.short_path = short_path
        self.long_path = long_path
        self.dynamic_path = dynamic_path

        self.short_term = {}
        self.long_term = {}
        self.dynamic = {}

        self.load_memory()


    def get_context_for_ai(self):
        """Converts JSON memory into understandable text for LLM"""
        # Take only important facts from user profile
        user_name = self.long_term.get("user_profile", {}).get("name", "User")
        
        # Save short block knowledge
        context = f"Information about user: Name is {user_name}.\n"
        
        # Take last 5 observations (to avoid bloating context)
        if self.dynamic.get("observations"):
            last_obs = self.dynamic["observations"][-5:]
            context += f"Recent observations: {'; '.join(last_obs)}.\n"
            
        return context


    # ---- JSON Load/Save ----

    def _load_json(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def _save_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # ---- Load all ----

    def load_memory(self):
        self.short_term = self._load_json(self.short_path) or {
            "conversation_context": [],
            "current_task": None,
            "assistant_mood": "neutral"
        }
        self.long_term = self._load_json(self.long_path) or {
            "user_profile": {},
            "assistant_rules": []
        }
        self.dynamic = self._load_json(self.dynamic_path) or {
            "observations": [],
            "learned_patterns": []
        }

    # ---- Save all ----

    def save_memory(self):
        self._save_json(self.short_path, self.short_term)
        self._save_json(self.long_path, self.long_term)
        self._save_json(self.dynamic_path, self.dynamic)

    # ---- Update short-term ----

    def update_short_term(self, user_msg, assistant_msg):
        self.short_term["conversation_context"].append({
            "user": user_msg,
            "assistant": assistant_msg
        })
        # Keep last 30 messages
        self.short_term["conversation_context"] = \
            self.short_term["conversation_context"][-30:]
        self.save_memory()

    # ---- Update long-term ----

    def update_long_term(self, text):
        # EXAMPLE: detect if user shares name
        t = text.lower()
        if "my name is" in t:
            name = text.split("my name is")[-1].strip()
            self.long_term.setdefault("user_profile", {})
            self.long_term["user_profile"]["name"] = name
            self.save_memory()

    # ---- Update dynamic ----

    def update_dynamic(self, text):
        self.dynamic["observations"].append(text)
        self.dynamic["observations"] = self.dynamic["observations"][-50:]
        self.save_memory()