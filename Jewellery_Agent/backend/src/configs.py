from src.config_nodes import AttributeConfig
from src.utils_db import get_unique_values

availabel_styles = get_unique_values("style")
availabel_materials = get_unique_values("material")

STYLE_LOGIC_MAP = {
    "Solitaire": {
        "vibe": "Timeless, elegant, minimalist, traditional, focus on the center stone.",
        "personas": ["Traditionalist", "Purist", "Minimalist", "Someone who values quality over flash"],
        "occasions": ["Classic Engagement", "Formal Event", "Daily Wear"]
    },
    "Halo": {
        "vibe": "Glamorous, flashy, maximum sparkle, makes the stone look bigger.",
        "personas": ["Trendsetter", "Glamorous", "Someone who likes attention", "Budget-conscious (wanting big look)"],
        "occasions": ["Romantic Proposal", "Cocktail Party", "Red Carpet"]
    },
    "Vintage": {
        "vibe": "Intricate, antique, detailed, romantic, warm, nostalgic.",
        "personas": ["Artist", "Historian", "Romantic", "Bohemian", "Dress Designer (appreciates detail)"],
        "occasions": ["Anniversary", "Heirloom Gift", "Artistic Gathering"]
    },
    "Pave": {  # or French Pavé
        "vibe": "Sophisticated, continuous sparkle, delicate, refined.",
        "personas": ["Fashionista", "Elegant", "Detail-oriented", "Modern Chic"],
        "occasions": ["Luxury Gift", "Wedding Band", "Upscale Dinner"]
    },
    "Twist": {
        "vibe": "Unique, fluid, artistic, modern, movement.",
        "personas": ["Designer", "Creative Director", "Architect", "Non-traditionalist"],
        "occasions": ["Creative Award", "Modern Engagement", "Fashion Statement"]
    },
    "Three-Stone": {
        "vibe": "Symbolic (Past/Present/Future), balanced, substantial.",
        "personas": ["Sentimental", "Romantic", "Mature", "Storyteller"],
        "occasions": ["Anniversary", "Relationship Milestone", "Renewal of Vows"]
    },
    "Bezel": {
        "vibe": "Sleek, modern, protective, snag-free, geometric.",
        "personas": ["Doctor", "Athlete", "Teacher", "Active Lifestyle", "Modernist"],
        "occasions": ["Active Daily Wear", "Travel Ring"]
    }
}

MATERIAL_LOGIC_MAP = {
    "White Gold": {
        "vibe": "Bright, modern, sleek, classic bridal look, mirrors the brilliance of diamonds.",
        "personas": ["Classic Modernist", "Versatile", "Cool-tone preference"],
        "pros": "Brightest white finish (due to rhodium), very popular."
    },
    "14k White Gold": {
        "vibe": "Durable, practical, identical look to standard white gold.",
        "personas": ["Practical", "Budget-conscious but quality-focused"],
        "pros": "Harder than 18k, good for active hands."
    },
    "Platinum": {
        "vibe": "Luxurious, prestigious, heavy/substantial feel, eternal.",
        "personas": ["Luxury Seeker", "Purist", "Someone with sensitive skin (Hypoallergenic)", "Nurse/Doctor (Hygiene)"],
        "pros": "Naturally white, doesn't fade, hypoallergenic, heaviest metal."
    },
    "Yellow Gold": {
        "vibe": "Traditional, warm, rich, vintage appeal, high contrast with diamonds.",
        "personas": ["Traditionalist", "Vintage Lover", "Warm-tone preference"],
        "pros": "Classic status, requires less maintenance than white gold."
    },
    "Rose Gold": {
        "vibe": "Romantic, soft, feminine, non-traditional, unique.",
        "personas": ["Trendsetter", "Romantic", "Bohemian", "Alternative"],
        "pros": "Very durable (copper alloy), distinct look."
    }
}

style_config = AttributeConfig(
    name="style",
    state_key="style",
    dependency_keys=[],
    valid_options=availabel_styles,
    prompt_template=f"""
    ### KNOWLEDGE BASE (Style Associations):
    {STYLE_LOGIC_MAP}

    ### INSTRUCTIONS:
    You are an expert Stylist. Your goal is to recommend the best jewelry style by synthesizing ALL clues from the conversation.

    **1. CHECK FOR DIRECT PREFERENCES (Override)**
    - If the user explicitly names a style (e.g., "I want a Halo") or describes a specific visual feature (e.g., "diamonds surrounding the center"), that takes precedence over everything else.

    **2. SYNTHESIZE CONTEXT (The "Stylist" Reasoning)**
    - If no direct style is named, combine cues from **Who** (Persona), **Where** (Occasion), and **What** (Vibe).
    - Look for the intersection of these clues.
    - *Example:* "A Dress Designer (Artistic) + Engagement (Romantic)" -> Might suggest **Vintage** or **Twist**.
    - *Example:* "A Nurse (Practical) + Daily Wear (Durable)" -> Might suggest **Bezel**.
    - *Example:* "A Lawyer (Professional) + Cocktail Party (Flashy)" -> Might suggest **Halo** or **Emerald Cut**.

    **3. MAPPING RULES**
    - Use the 'KNOWLEDGE BASE' above as a guide to match user descriptions to our inventory.
    - **Multiple Matches:** If the context fits multiple styles equally well, return the top 2-3 most relevant options.
    - **Unsure?** If the information is too vague to make a professional recommendation (e.g., just "I need a ring"), return ["None"]. Do not guess randomly.
    - But if the user think any style is good, or said I want to try any options, return all valid options.

    **4. EXPLAIN YOUR LOGIC**
    - In the 'reasoning' field, explain the synthesis: "Recommended [Styles] because the user mentioned [Persona/Event], which aligns with [Style Qualities]."
    """
)

material_config = AttributeConfig(
    name="material",
    state_key="material",
    dependency_keys=["style"],
    valid_options=availabel_materials,
    prompt_template=f"""
    ### KNOWLEDGE BASE (Material Associations):
    {MATERIAL_LOGIC_MAP}

    ### INSTRUCTIONS:
    You are an expert Jeweler. Your goal is to recommend the metal/material based on the user's lifestyle, aesthetic, and previously selected style.

    **1. CHECK FOR DIRECT PREFERENCES (Override)**
    - If the user explicitly asks for "Gold" (implies Yellow), "Silver/White" (implies White Gold or Platinum), or a specific metal, that takes precedence.
    - Note: If user says "Silver color", map to White Gold or Platinum based on other clues (budget/durability).

    **2. ANALYZE LIFESTYLE & DURABILITY**
    - **Active/Heavy Use:** If the user has an active job (Nurse, Chef, Athlete), recommend **Platinum** (most secure prongs, durable) or **14k White Gold** (harder alloy).
    - **Sensitive Skin:** Recommend **Platinum** (Hypoallergenic).

    **3. CROSS-REFERENCE WITH STYLE**
    - Look at the 'style' provided in the context.
    - **Vintage Styles:** Strongly correlated with **Yellow Gold** or **Rose Gold**.
    - **Modern/Solitaire/Halo:** Strongly correlated with **White Gold** or **Platinum**.
    - **Twist/Unique:** Often seen in **Rose Gold** or Mixed metals.

    **4. AESTHETICS & SKIN TONE**
    - **Warm Tones:** Yellow Gold, Rose Gold.
    - **Cool Tones:** White Gold, Platinum.
    - **Mixed:** Rose Gold acts as a great neutral.

    **5. EXPLAIN YOUR LOGIC**
    - Explain: "Recommended [Material] because the user prefers a [Style] look and has [Lifestyle/Skin Tone needs]."
    - If distinguishing between Platinum and White Gold, cite the 'heft' or 'hypoallergenic' nature of Platinum vs the 'brightness' of White Gold.
    - If the user think any style is good, or said I want to try any options, return all valid options.
    
    You msut be very careful with very similar material. For example, "White gold" and "14K White gold". Don't be confused.
    """
)
