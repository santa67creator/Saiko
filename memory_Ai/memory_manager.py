import json
import os

class MemoryManager:
    def __init__(self, short_path="short_memory.json",
                 long_path="long_memory.json",
                 dynamic_path="dynamic_memory.json"
                 ):
        
        self.run_count = 0
        self.short_path = short_path
        self.long_path = long_path
        self.dynamic_path = dynamic_path

        self.short_term = {}
        self.long_term = {}
        self.dynamic = {}

        self.load_memory()


    def get_context_for_ai(self):
        # Take only important facts from user profile
        user_name = self.long_term.get("user_profile", {}).get("name", "User")
        context = f"Information about user: Name is {user_name}.\n"

        # Save short block knowledge
        important_info = self.long_term.get("facts", {}).get("important_info", [])
        if important_info:
            context += f"Important facts about user: {'; '.join(important_info[-5:])}.\n"
        
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
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"Error saving {path}: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


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
        self.run_count = self.long_term.get("run_count", 0)

    # ---- Save all ----

    def save_memory(self):
        self.long_term["run_count"] = self.run_count
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

    # ---- Update long-term ----

    def update_long_term(self, text):
        # EXAMPLE: detect if user shares name
        t = text.lower()
        if "my name is" in t:
            name = text.split("my name is")[-1].strip().split()[0].strip(".,!?")
            if len(name) > 1 and name != "...":
                self.long_term.setdefault("user_profile", {})
                self.long_term["user_profile"]["name"] = name

        important_patterns = ["i like", "i dislike", "my favorite", "i want", "i love", "my hobby", "i prefer",
        "i hate", "i don't like", "my goal", "i want to be",
        "i like to", "i study", "i am learning", "i enjoy"]

        for pattern in important_patterns:
            if pattern in t:
                self.long_term.setdefault("facts", {})
                self.long_term["facts"].setdefault("important_info", [])
                self.long_term["facts"]["important_info"].append(text)
                # Keep only last 20 important info pieces
                self.long_term["facts"]["important_info"] = \
                    self.long_term["facts"]["important_info"][-50:]
                break

        self.run_count += 1

        if self.run_count % 5 == 0:  # Every 5 runs, trim memory
            self.trim_memory()
            print("Memory trimmed to prevent overload.")


    # ---- Update dynamic ----

    def update_dynamic(self, text):
        noise = {"[blank_audio]", "(popping)", "(clicking)", "."}
        if len(text.strip()) < 4 or text.strip().lower() in noise:
            return
        self.dynamic["observations"].append(text)
        self.dynamic["observations"] = self.dynamic["observations"][-50:]



    def trim_memory(self):
        """Auto-clean memory to prevent overload."""


        if "facts" in self.long_term and "important_info" in self.long_term["facts"]:
            facts = self.long_term["facts"]["important_info"]
            cleaned_info = []

            
            for info in facts:
                if not info or info.strip() == "":  # Skip empty facts
                    continue

                if len(info) > 250:  # Limit to 250 characters
                    info = info[:200] + "..."  # Truncate long facts
                cleaned_info.append(info)



            seen = set()
            unique_info = []
            for info in cleaned_info:
                if info not in seen:
                    unique_info.append(info)
                    seen.add(info)


            self.long_term["facts"]["important_info"] = unique_info[-100:]  # Final limit to 100 unique important facts
            
        # CLEAN DYNAMIC MEMORY OBS
        if "observations" in self.dynamic:
            obs = self.dynamic["observations"]
            cleaned_obs = []
            for o in obs:
                if o and o.strip() != "":
                    cleaned_obs.append(o)

            # Remove duplicates while preserving order
            unique_obs = []
            seen_obs = set()
            for o in cleaned_obs:
                if o not in seen_obs:
                    unique_obs.append(o)
                    seen_obs.add(o)
    
            self.dynamic["observations"] = unique_obs[-50:]  # Keep last 50 unique observations

        # This can be called periodically to trim memory if it grows too large
        # CLEAN SHORT-TERM MEMORY
        if "conversation_context" in self.short_term:

            ctx = self.short_term["conversation_context"]
            #remove empty messages or invalid entries
            
            cleaned_ctx = [ 
                m for m in ctx
                if m.get("user") or m.get("assistant")  # Keep if either user or assistant message exists
            ]

            self.short_term["conversation_context"] = cleaned_ctx[-30:]
