## 📍 Step 1: Scraping Posts Using a Browser Snippet

We used a **browser-based JavaScript snippet** to extract posts from any Discourse thread.

### ✅ How to Use:

1. Open your browser (Chrome, Edge, Firefox).
2. Go to any Discourse thread (e.g., IITM TDS forum).
3. Open the **DevTools Console** (`Ctrl+Shift+J`).
4. Paste and run this code:

```js
(() => {
  const threadTitle = document.querySelector('title')?.innerText.trim();
  const threadUrl = window.location.href;

  const posts = [...document.querySelectorAll('.topic-post')].map(post => {
    const content = post.querySelector('.cooked')?.innerText.trim() || "";
    const author = post.querySelector('.names .first')?.innerText.trim() || "unknown";
    const time = post.querySelector('time')?.getAttribute('datetime');
    return { author, time, content };
  });

  const threadData = {
    title: threadTitle,
    url: threadUrl,
    posts
  };

  console.log("✅ Thread parsed:");
  console.log(JSON.stringify(threadData, null, 2));
})();
```

### 📦 Output:

* JSON object printed to console
* Contains:

  * `title`: thread title
  * `url`: thread URL
  * `posts`: list of `{ author, time, content }`

---

## 📍 Step 2: Filtering Threads by Date Range

Once you gather multiple thread JSONs into a single file (e.g., `tds_discourse_posts.json`), we use a Python script to **filter only relevant posts** between Jan 1 and Apr 14, 2025.

### 🔍 Python Script:

```python
import json
from datetime import datetime

# Load scraped threads
with open("tds_discourse_posts.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

# Define date range
start = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
end = datetime.fromisoformat("2025-04-14T23:59:59+00:00")

def in_range(post_time):
    dt = datetime.fromisoformat(post_time)
    return start <= dt <= end

# Filter posts
filtered_threads = []
for thread in threads:
    filtered_posts = [p for p in thread.get("posts", []) if in_range(p["time"])]
    if filtered_posts:
        filtered_threads.append({
            "title": thread["title"],
            "url": thread["url"],
            "posts": filtered_posts
        })

# Save to file
with open("tds_discourse_posts_filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered_threads, f, indent=2, ensure_ascii=False)

print(f"✅ Done. Filtered {len(filtered_threads)} threads.")
```

### 📂 Output:

* A new file: `tds_discourse_posts_filtered.json`
* Contains only threads with at least 1 post within the date range
* Ready for use in RAG (Retrieval-Augmented Generation) or embedding pipelines

---

## 💡 Notes

* This method avoids using the Discourse API and works without authentication.
* Useful for quickly extracting and cleaning TDS-related Q\&A from forums.
* Easily adaptable for any course that uses Discourse!

---
