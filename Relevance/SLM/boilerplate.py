clip_before = [
    "You have a preview view of this article while we are checking your access. When we have confirmed access, the full article content will load."
    "Share All sharing options for:",
    """
    Now playing

    - Source: CNN
    """,
    "Act on the news with POLITICO Pro.",
    "Video Ad Feedback",
    "Read the automated transcript below. Because it’s a computer-generated transcript, it contains many errors and misspellings.",
    "A lot of things happened. Here are some of the things. This is TPM’s Morning Memo.",
    "Good morning, Chicago.",
]
clip_after = [
    "Support HuffPost",
    "Sign up for the newsletter Today, Explained",
    "RELATED STORIES",
    "Editor’s note: FactCheck.org does not accept advertising",
    "Get the How to Do It Newsletter",
    "Thanks for signing up! You can manage your newsletter subscriptions at any time.",
    "More Advice From Slate",
    "Start your day right",
    "Get Essential San Diego, weekday mornings",
    "In This Stream",
    "This article originally appeared in The New York Times."
    "Follow NBC Out",
    "It takes longer to read this sentence than it does to support our work.",
    "To continue reading this article...",
    "About NEWSMAX TV:",
    "Newsletter Signup",
    "Today's reads",
    "This article contains affiliate links; if you click such a link and make a purchase, we may earn a commission.",
    "What We're Reading",
    "Thank you for your patience while we verify access.",
    "Health reads",
    "Random reads",
    "Our journalism needs your support.",
    "Read more on MLive:",
    "Send questions/comments to the editors.",
    "See more content from me:",
    "More top news stories:",
    "MORE ON:"
    "Click here to see our full coverage of the coronavirus outbreak.",
    "Further Reading",
    "Be the first to know",
]
clip_at = [
    "The Washington PostDemocracy Dies in Darkness",
    "Advertisement",
    "Read the Full Transcript",
    "Advertisement:",
    "Notice: Transcripts are machine and human generated and lightly edited for accuracy. They may contain errors.",
    "CLICK HERE TO GET THE FOX NEWS APP",
    "SKIP ADVERTISEMENT",
    "You have a preview view of this article while we are checking your access. When we have confirmed access, the full article content will load."
]

def filter_boilerplate(txt):
    for phrase in clip_before:
        txt = txt.split(phrase)[-1]
    for phrase in clip_after:
        txt = txt.split(phrase)[0]
    for phrase in clip_at:
        txt = txt.replace(phrase, '')
    return txt