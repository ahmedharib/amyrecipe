"""Build a structured >=250-recipe corpus (grounded in public HuggingFace recipes + curated variations)
and a >=45-query held-out retrieval test set with ground-truth relevant IDs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.vision.dataset import SELECTED_CLASSES

# Ingredient keywords for deterministic diet tagging
_MEAT_FISH_KEYWORDS = {
    "beef", "chicken", "pork", "bacon", "ham", "turkey", "lamb", "sausage",
    "pepperoni", "prosciutto", "salami", "pancetta", "steak", "ground beef",
    "shrimp", "prawn", "crab", "lobster", "salmon", "tuna", "fish", "anchovy",
    "anchovies", "cod", "tilapia", "duck", "venison", "meatball", "chashu",
}
_DAIRY_EGG_HONEY_KEYWORDS = {
    "milk", "cheese", "cheddar", "mozzarella", "parmesan", "ricotta", "butter",
    "cream", "yogurt", "whey", "ghee", "egg", "eggs", "mayonnaise", "honey",
    "feta", "brie", "mascarpone", "pecorino", "buttermilk",
}
_GLUTEN_KEYWORDS = {
    "wheat", "all purpose flour", "whole wheat flour", "spelt flour", "bread", "bun", "brioche bun",
    "pasta", "ramen noodles", "egg noodles", "spaghetti", "lasagna noodles", "lasagna sheets",
    "flour tortilla", "pizza dough", "crouton", "croutons", "soy sauce", "panko", "breadcrumb",
    "breadcrumbs", "barley", "rye",
}
_SPICY_KEYWORDS = {
    "chili", "chiles", "jalapeno", "sriracha", "cayenne", "pepper flakes",
    "habanero", "gochujang", "spicy", "hot sauce", "chipotle", "red pepper",
}
_HIGH_PROTEIN_KEYWORDS = {
    "chicken", "beef", "salmon", "tuna", "tofu", "tempeh", "edamame", "egg",
    "eggs", "lentil", "lentils", "chickpea", "chickpeas", "black beans",
    "greek yogurt", "shrimp", "turkey", "seitan",
}

_CLASS_MATCHERS = {
    "caesar_salad": ["caesar salad", "chicken caesar", "kale caesar"],
    "chocolate_cake": ["chocolate cake", "chocolate lava cake", "fudge cake", "flourless chocolate cake"],
    "hamburger": ["hamburger", "cheeseburger", "veggie burger", "black bean burger", "turkey burger"],
    "lasagna": ["lasagna", "lasagne", "zucchini lasagna", "vegetable lasagna"],
    "pad_thai": ["pad thai", "tofu pad thai", "shrimp pad thai"],
    "pancakes": ["pancakes", "pancake", "buttermilk pancakes", "oatmeal pancakes", "banana pancakes"],
    "pizza": ["pizza", "margherita pizza", "pepperoni pizza", "cauliflower crust pizza"],
    "ramen": ["ramen", "miso ramen", "shoyu ramen", "tonkotsu ramen", "spicy ramen"],
    "sushi": ["sushi", "sushi roll", "california roll", "avocado roll", "salmon sushi"],
    "tacos": ["tacos", "taco", "fish tacos", "black bean tacos", "chicken tacos", "beef tacos"],
}


def infer_diet_tags(title: str, ingredients: List[str]) -> List[str]:
    blob = f"{title} {' '.join(ingredients)}".lower()
    words = set(re.findall(r"[a-z]+", blob))

    def has_any(keywords: set[str]) -> bool:
        for kw in keywords:
            if " " in kw:
                if kw in blob:
                    return True
            elif kw in words:
                return True
        return False

    has_meat = has_any(_MEAT_FISH_KEYWORDS)
    has_dairy_egg = has_any(_DAIRY_EGG_HONEY_KEYWORDS)
    has_gluten = has_any(_GLUTEN_KEYWORDS)
    is_spicy = has_any(_SPICY_KEYWORDS)
    is_high_prot = has_any(_HIGH_PROTEIN_KEYWORDS)

    tags: List[str] = []
    if not has_meat:
        tags.append("vegetarian")
        if not has_dairy_egg:
            tags.append("vegan")
    if not has_dairy_egg:
        tags.append("dairy_free")
    if not has_gluten:
        tags.append("gluten_free")
    if is_spicy:
        tags.append("spicy")
    if is_high_prot:
        tags.append("high_protein")
    return sorted(tags)


def _clean_title(title: str) -> str:
    title = title.strip().strip('"\'').strip()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"([A-Za-z0-9])\(", r"\1 (", title)
    title = re.sub(r"'S\b", "'s", title)
    return title


def _clean_ingredient_preview(ing: str) -> str:
    """Strip leading measurement quantities for concise recipe descriptions."""
    text = re.sub(
        r"^(?:\d+\s+)?(?:\d+/\d+|\d+(?:\.\d+)?)\s*(?:\([^)]*\)\s*)?(?:c\.|cup|cups|tbsp\.?|tsp\.?|oz\.?|lb\.?|lbs\.?|pkg\.?|can|cans|jar|jars|cloves?|slices?|pieces?)?\s*",
        "",
        ing,
        flags=re.I,
    ).strip()
    return text or ing


def fetch_public_hf_recipes(target_per_class: int = 23, max_scan: int = 60000) -> Dict[str, List[Dict[str, Any]]]:
    """Stream real recipes from public Hugging Face dataset `corbt/all-recipes`."""
    collected: Dict[str, List[Dict[str, Any]]] = {c: [] for c in SELECTED_CLASSES}
    seen_titles: set[str] = set()
    try:
        from datasets import load_dataset

        ds = load_dataset("corbt/all-recipes", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            if all(len(v) >= target_per_class for v in collected.values()):
                break

            raw_text = str(row.get("input", ""))
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            if len(lines) < 3:
                continue
            title = _clean_title(lines[0])
            title_lower = title.lower()
            if title_lower in seen_titles or len(title) > 75:
                continue
            if any(bad in title_lower for bad in ("vinegar for", "dressing", "seasoning", "sauce mix", "spice mix")):
                continue

            matched_cls = None
            for cls_name, kws in _CLASS_MATCHERS.items():
                if len(collected[cls_name]) < target_per_class and any(kw in title_lower for kw in kws):
                    matched_cls = cls_name
                    break
            if not matched_cls:
                continue

            ings: List[str] = []
            steps: List[str] = []
            mode = None
            for ln in lines[1:]:
                low = ln.lower()
                if low.startswith("ingredients"):
                    mode = "ing"
                    continue
                if low.startswith("directions") or low.startswith("instructions"):
                    mode = "step"
                    continue
                is_bullet = bool(re.match(r"^\s*[-*•]\s*", ln))
                clean = re.sub(r"\s+", " ", re.sub(r"^\s*[-*•]\s*", "", ln)).strip()
                if not clean:
                    continue
                if mode == "ing":
                    if is_bullet or not ings:
                        ings.append(clean)
                    else:
                        ings[-1] = f"{ings[-1]} {clean}"
                elif mode == "step":
                    prev_incomplete = bool(steps and not re.search(r"[.!?]$", steps[-1]))
                    starts_lower = bool(clean and clean[0].islower())
                    if not steps or (is_bullet and not (prev_incomplete and starts_lower)):
                        steps.append(clean)
                    else:
                        steps[-1] = f"{steps[-1]} {clean}"

            if len(ings) < 3 or not steps:
                continue
            seen_titles.add(title_lower)
            prep_time = min(75, max(12, 10 + 5 * len(steps)))
            diet_tags = infer_diet_tags(title, ings)
            preview_ings = [_clean_ingredient_preview(x) for x in ings[:4]]
            collected[matched_cls].append({
                "title": title,
                "dish_category": matched_cls,
                "ingredients": ings[:12],
                "steps": steps[:8],
                "prep_time_min": prep_time,
                "diet_tags": diet_tags,
                "description": f"{title} with {', '.join(preview_ings)}.",
                "source": "corbt/all-recipes (Hugging Face)",
            })
    except Exception as exc:
        print(f"[build_recipes] Public HF stream warning: {exc}")
    return collected


def _curated_supplements() -> List[Dict[str, Any]]:
    """Curated, realistic recipes guaranteeing rich coverage across diet/time constraints for all 10 classes."""
    templates = [
        # caesar_salad
        ("Classic Grilled Chicken Caesar Salad", "caesar_salad", 20, ["romaine lettuce", "grilled chicken breast", "parmesan", "garlic", "lemon juice", "olive oil", "sourdough croutons"], "Crisp romaine tossed in tangy dressing with grilled chicken."),
        ("Quick 15-Minute Kale Caesar Salad", "caesar_salad", 15, ["baby kale", "romaine", "parmesan", "olive oil", "lemon", "dijon mustard", "gluten free croutons"], "Fast vegetarian kale and romaine caesar salad."),
        ("Vegan Avocado Chickpea Caesar Salad", "caesar_salad", 15, ["romaine lettuce", "roasted chickpeas", "avocado", "tahini", "lemon juice", "nutritional yeast", "capers"], "Plant-based creamy tahini caesar with protein-rich chickpeas."),
        ("Spicy Blackened Shrimp Caesar Bowl", "caesar_salad", 20, ["romaine", "shrimp", "cayenne", "garlic", "olive oil", "lemon", "parmesan"], "Zesty blackened shrimp over crisp caesar greens."),
        ("Gluten-Free Crispy Tofu Caesar Salad", "caesar_salad", 22, ["romaine", "firm tofu", "tamari", "lemon", "tahini", "olive oil", "garlic"], "High-protein vegan and gluten-free caesar bowl."),
        ("Mediterranean Yogurt Caesar Salad", "caesar_salad", 15, ["romaine", "greek yogurt", "lemon", "garlic", "olive oil", "roasted chickpeas"], "Lighter protein-packed yogurt dressing caesar without gluten."),

        # chocolate_cake
        ("Flourless Dark Chocolate Almond Cake", "chocolate_cake", 35, ["dark chocolate", "almond flour", "eggs", "cocoa powder", "espresso", "coconut oil"], "Rich gluten-free and dairy-free dark chocolate cake."),
        ("15-Minute Vegan Chocolate Mug Cake", "chocolate_cake", 15, ["oat flour", "cocoa powder", "almond milk", "maple syrup", "coconut oil", "dark chocolate chips"], "Single-pan quick plant-based and gluten-free chocolate treat."),
        ("Classic Double Chocolate Layer Cake", "chocolate_cake", 55, ["all purpose flour", "cocoa powder", "butter", "eggs", "milk", "sugar", "dark chocolate"], "Moist bakery-style chocolate fudge cake."),
        ("Spicy Mexican Cinnamon Chocolate Cake", "chocolate_cake", 40, ["almond flour", "cocoa powder", "cinnamon", "cayenne", "maple syrup", "coconut milk"], "Warm spiced vegan and gluten-free chocolate cake."),
        ("Quick Zucchini Chocolate Skillet Cake", "chocolate_cake", 25, ["whole wheat flour", "cocoa powder", "shredded zucchini", "oat milk", "olive oil", "sugar"], "Tender vegetarian and dairy-free chocolate cake."),
        ("High-Protein Greek Yogurt Chocolate Cake", "chocolate_cake", 30, ["oat flour", "greek yogurt", "cocoa powder", "eggs", "honey", "dark chocolate"], "Wholesome high-protein chocolate cake."),

        # hamburger
        ("Classic Smash Beef Cheeseburger", "hamburger", 18, ["ground beef", "cheddar", "brioche bun", "onion", "pickles", "mustard"], "Crispy-edged diner smash burger in under 20 minutes."),
        ("Spicy Black Bean Vegan Burger", "hamburger", 22, ["black beans", "oats", "chipotle", "red onion", "garlic", "lettuce", "tomato", "gluten free bun"], "Hearty plant-based smoky bean patty without dairy or gluten."),
        ("Crispy Portobello Mushroom Burger", "hamburger", 20, ["portobello mushroom", "olive oil", "balsamic vinegar", "garlic", "lettuce", "tomato", "avocado"], "Savory vegetarian and dairy-free burger."),
        ("Lean Turkey Avocado Lettuce-Wrap Burger", "hamburger", 18, ["ground turkey", "avocado", "butter lettuce", "tomato", "garlic", "cumin"], "Low-carb, gluten-free, dairy-free high-protein burger."),
        ("Crispy Quinoa Chickpea Veggie Burger", "hamburger", 25, ["chickpeas", "quinoa", "carrot", "cumin", "spinach", "olive oil"], "Wholesome vegan protein burger."),
        ("Jalapeno Pepper Jack Beef Burger", "hamburger", 20, ["ground beef", "jalapeno", "pepper jack cheese", "brioche bun", "chili powder"], "Fiery spicy burger ready in 20 minutes."),

        # lasagna
        ("Classic Beef & Ricotta Lasagna", "lasagna", 65, ["lasagna noodles", "ground beef", "ricotta", "mozzarella", "tomato sauce", "garlic", "basil"], "Rich baked Italian meat lasagna."),
        ("Roasted Vegetable Spinach Lasagna", "lasagna", 50, ["lasagna sheets", "spinach", "zucchini", "ricotta", "mozzarella", "marinara"], "Hearty vegetarian three-cheese veggie lasagna."),
        ("Gluten-Free Zucchini Ribbon Lasagna", "lasagna", 45, ["zucchini", "ground turkey", "tomato sauce", "basil", "olive oil", "garlic"], "Light high-protein lasagna using zucchini instead of pasta."),
        ("Vegan Lentil Cashew Cream Lasagna", "lasagna", 50, ["brown lentils", "zucchini", "cashews", "nutritional yeast", "marinara", "spinach", "garlic"], "Plant-based dairy-free and gluten-free lentil lasagna."),
        ("Quick 25-Minute Skillet Tofu Lasagna", "lasagna", 25, ["gluten free pasta", "crumbled tofu", "marinara", "spinach", "basil", "olive oil"], "Fast weeknight vegan skillet lasagna."),
        ("Spicy Arrabbiata Sausage Lasagna", "lasagna", 55, ["lasagna noodles", "italian sausage", "chili flakes", "mozzarella", "tomato sauce"], "Bold spicy baked lasagna."),

        # pad_thai
        ("15-Minute Crispy Tofu Pad Thai", "pad_thai", 15, ["rice noodles", "firm tofu", "tamarind paste", "tamari", "peanuts", "bean sprouts", "lime"], "Quick authentic vegan and gluten-free pad thai."),
        ("Spicy Shrimp & Garlic Pad Thai", "pad_thai", 20, ["rice noodles", "shrimp", "chili flakes", "garlic", "fish sauce", "lime", "peanuts", "egg"], "Savory high-protein street-style pad thai."),
        ("Classic Chicken Pad Thai", "pad_thai", 22, ["rice noodles", "chicken breast", "tamarind", "egg", "green onions", "peanuts", "lime"], "Balanced sweet-tangy gluten-free chicken pad thai."),
        ("Plant-Based Peanut Veggie Pad Thai", "pad_thai", 18, ["rice noodles", "broccoli", "carrot", "peanut butter", "tamari", "lime", "sriracha"], "Dairy-free spicy vegan noodle stir-fry."),
        ("Extra-Spicy Chili Garlic Tofu Pad Thai", "pad_thai", 20, ["rice noodles", "tofu", "sriracha", "garlic", "tamari", "peanuts", "scallions"], "Fiery meatless pad thai ready in 20 minutes."),
        ("Low-Carb Zucchini Noodle Pad Thai", "pad_thai", 15, ["zucchini noodles", "shrimp", "tamarind", "lime", "peanuts", "chili"], "Ultra-fast gluten-free and dairy-free pad thai."),

        # pancakes
        ("Fluffy Classic Buttermilk Pancakes", "pancakes", 18, ["flour", "buttermilk", "egg", "butter", "baking powder", "maple syrup"], "Golden diner-style breakfast pancakes."),
        ("15-Minute Banana Oat Blender Pancakes", "pancakes", 15, ["rolled oats", "banana", "almond milk", "cinnamon", "baking powder", "maple syrup"], "Quick vegan, dairy-free, and gluten-free pancakes."),
        ("High-Protein Cottage Cheese Pancakes", "pancakes", 15, ["rolled oats", "eggs", "greek yogurt", "vanilla", "berries"], "Protein-packed gluten-free breakfast stack."),
        ("Vegan Blueberry Buckwheat Pancakes", "pancakes", 20, ["buckwheat flour", "oat milk", "blueberries", "coconut oil", "maple syrup"], "Wholesome plant-based gluten-free berry pancakes."),
        ("Dairy-Free Lemon Ricotta-Style Almond Pancakes", "pancakes", 20, ["almond flour", "coconut milk", "lemon zest", "eggs", "maple syrup"], "Light citrus gluten-free and dairy-free pancakes."),
        ("Cinnamon Apple Spelt Pancakes", "pancakes", 22, ["spelt flour", "apple", "oat milk", "cinnamon", "maple syrup"], "Warm spiced vegan pancakes."),

        # pizza
        ("Quick 20-Minute Margherita Flatbread Pizza", "pizza", 20, ["pizza dough", "tomato sauce", "fresh mozzarella", "basil", "olive oil"], "Classic vegetarian stone-baked style pizza."),
        ("Gluten-Free Cauliflower Crust Veggie Pizza", "pizza", 30, ["cauliflower", "almond flour", "tomato sauce", "bell pepper", "mushrooms", "spinach", "olive oil"], "Plant-based dairy-free and gluten-free crisp pizza."),
        ("Classic Pepperoni & Hot Honey Pizza", "pizza", 25, ["pizza dough", "pepperoni", "mozzarella", "tomato sauce", "chili flakes", "hot sauce"], "Spicy crispy pepperoni pizza."),
        ("Vegan Roasted Mushroom Truffle Pizza", "pizza", 22, ["pizza dough", "cremini mushrooms", "garlic", "olive oil", "arugula", "nutritional yeast"], "Savory dairy-free vegan pizza ready in 22 minutes."),
        ("High-Protein Chickpea Crust Pesto Pizza", "pizza", 25, ["chickpea flour", "basil pesto", "cherry tomatoes", "spinach", "olive oil"], "Quick gluten-free and vegan protein pizza."),
        ("Spicy Buffalo Chickpea Pizza", "pizza", 20, ["gluten free crust", "chickpeas", "hot sauce", "red onion", "olive oil", "cilantro"], "Fiery meatless pizza under 25 minutes."),

        # ramen
        ("15-Minute Spicy Miso Shiitake Ramen", "ramen", 15, ["ramen noodles", "white miso", "shiitake mushrooms", "sriracha", "garlic", "ginger", "bok choy"], "Fast umami-rich vegan and dairy-free noodle soup."),
        ("Rich Chicken Shoyu Ramen Bowl", "ramen", 25, ["ramen noodles", "chicken broth", "chicken breast", "soy sauce", "soft boiled egg", "scallions"], "Savory high-protein Japanese noodle soup."),
        ("Gluten-Free Rice Noodle Coconut Curry Ramen", "ramen", 20, ["rice noodles", "coconut milk", "red curry paste", "tofu", "spinach", "lime", "tamari"], "Creamy dairy-free vegan and gluten-free ramen."),
        ("Spicy Chili Garlic Tofu Ramen", "ramen", 18, ["ramen noodles", "crispy tofu", "chili oil", "garlic", "miso", "sesame", "scallions"], "Fiery 18-minute plant-based ramen."),
        ("Quick Sesame Peanut Veggie Ramen", "ramen", 15, ["ramen noodles", "peanut butter", "miso", "garlic", "spinach", "chili flakes"], "Creamy vegan weeknight noodle bowl."),
        ("20-Minute Sesame Ginger Edamame Ramen", "ramen", 20, ["rice noodles", "edamame", "ginger", "garlic", "tamari", "sesame oil", "chili"], "Gluten-free, vegan, high-protein ramen."),

        # sushi
        ("Quick Avocado Cucumber Maki Sushi", "sushi", 20, ["sushi rice", "nori", "avocado", "cucumber", "rice vinegar", "tamari"], "Fresh vegan, dairy-free, and gluten-free sushi rolls."),
        ("Spicy Salmon & Avocado Sushi Bowl", "sushi", 15, ["sushi rice", "fresh salmon", "avocado", "cucumber", "sriracha", "nori", "tamari"], "Fast 15-minute high-protein deconstructed sushi."),
        ("Crispy Tofu & Mango Sushi Roll", "sushi", 25, ["sushi rice", "nori", "crispy tofu", "mango", "avocado", "sesame seeds"], "Sweet-savory plant-based and gluten-free roll."),
        ("High-Protein Spicy Tuna Poke Sushi", "sushi", 15, ["sushi rice", "ahi tuna", "sriracha", "sesame oil", "avocado", "edamame", "nori"], "Quick gluten-free and dairy-free spicy tuna bowl."),
        ("Roasted Sweet Potato Tempura-Style Roll", "sushi", 25, ["sushi rice", "nori", "sweet potato", "avocado", "tamari", "sesame"], "Hearty vegan and gluten-free sushi roll."),
        ("Classic California Crab & Avocado Roll", "sushi", 25, ["sushi rice", "nori", "crab", "avocado", "cucumber", "sesame seeds"], "Refreshing coastal sushi roll."),

        # tacos
        ("15-Minute Spicy Black Bean Tacos", "tacos", 15, ["corn tortillas", "black beans", "cumin", "chili powder", "avocado", "lime", "cilantro"], "Fast vegan, dairy-free, and gluten-free street tacos."),
        ("Crispy Baja Fish Tacos with Lime Slaw", "tacos", 20, ["corn tortillas", "cod", "cabbage", "lime", "cilantro", "jalapeno", "olive oil"], "Fresh coastal gluten-free and dairy-free fish tacos."),
        ("Smoky Chipotle Chicken Street Tacos", "tacos", 20, ["corn tortillas", "chicken breast", "chipotle", "onion", "cilantro", "lime"], "High-protein spicy gluten-free tacos."),
        ("Roasted Cauliflower & Chickpea Tacos", "tacos", 25, ["corn tortillas", "cauliflower", "chickpeas", "smoked paprika", "avocado", "lime"], "Hearty plant-based gluten-free tacos."),
        ("Spicy Korean-Style Tofu Tacos", "tacos", 18, ["corn tortillas", "firm tofu", "gochujang", "cabbage", "scallions", "sesame"], "Bold vegan gluten-free fusion tacos."),
        ("Classic Seasoned Beef Street Tacos", "tacos", 18, ["corn tortillas", "ground beef", "cumin", "garlic", "onion", "cilantro", "lime"], "Quick weeknight gluten-free and dairy-free tacos."),
    ]

    out: List[Dict[str, Any]] = []
    for title, cat, t_min, ings, desc in templates:
        out.append({
            "title": title,
            "dish_category": cat,
            "ingredients": ings,
            "steps": [
                f"Prepare and measure {', '.join(ings[:3])}.",
                f"Cook and combine remaining ingredients for about {t_min} minutes until ready.",
                "Plate warm and garnish to taste.",
            ],
            "prep_time_min": t_min,
            "diet_tags": infer_diet_tags(title, ings),
            "description": desc,
            "source": "Curated domain recipe supplement",
        })
    return out


def build_recipes_and_queries(out_dir: Path = Path("data")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_collected = fetch_public_hf_recipes(target_per_class=23, max_scan=60000)

    all_recipes: List[Dict[str, Any]] = []
    seen_titles: set[str] = set()
    for cls_name in SELECTED_CLASSES:
        for rec in hf_collected[cls_name]:
            t_low = rec["title"].lower()
            if t_low not in seen_titles:
                seen_titles.add(t_low)
                all_recipes.append(rec)

    for rec in _curated_supplements():
        t_low = rec["title"].lower()
        if t_low not in seen_titles:
            seen_titles.add(t_low)
            all_recipes.append(rec)

    # Assign deterministic IDs
    for idx, rec in enumerate(all_recipes, start=1):
        rec["id"] = f"rec_{idx:03d}"

    (out_dir / "recipes.json").write_text(json.dumps(all_recipes, indent=2), encoding="utf-8")

    # Build 45 diverse held-out test queries across all 10 classes + constraints + synonyms
    query_specs = [
        ("vegetarian pizza under 30 minutes", "pizza", ["vegetarian"], 30, []),
        ("vegan plant-based pizza dairy free", "pizza", ["vegan", "dairy_free"], None, []),
        ("gluten free pizza under 30 mins", "pizza", ["gluten_free"], 30, []),
        ("spicy pepperoni pizza", "pizza", ["spicy"], None, ["pepperoni"]),
        ("quick meatless noodle stir fry under 20 min", "pad_thai", ["vegetarian"], 20, ["tofu", "peanut", "noodles"]),
        ("spicy shrimp pad thai", "pad_thai", ["spicy"], None, ["shrimp"]),
        ("vegan gluten free pad thai", "pad_thai", ["vegan", "gluten_free"], None, []),
        ("quick pad thai in 20 minutes", "pad_thai", [], 20, []),
        ("quick vegan ramen under 20 minutes", "ramen", ["vegan"], 20, []),
        ("spicy plant based noodle soup", "ramen", ["vegan", "spicy"], None, []),
        ("gluten free dairy free ramen", "ramen", ["gluten_free", "dairy_free"], None, []),
        ("high protein chicken ramen bowl", "ramen", ["high_protein"], None, ["chicken"]),
        ("vegan gluten free sushi under 25 min", "sushi", ["vegan", "gluten_free"], 25, []),
        ("quick spicy tuna or salmon sushi bowl", "sushi", ["spicy", "high_protein"], 20, []),
        ("plant based avocado roll", "sushi", ["vegan"], None, ["avocado"]),
        ("gluten free sushi with tofu", "sushi", ["gluten_free"], None, ["tofu"]),
        ("vegetarian tacos under 20 minutes", "tacos", ["vegetarian"], 20, []),
        ("spicy vegan gluten free street tacos", "tacos", ["vegan", "gluten_free", "spicy"], None, []),
        ("high protein chicken tacos under 25 mins", "tacos", ["high_protein"], 25, ["chicken"]),
        ("dairy free fish tacos", "tacos", ["dairy_free"], None, ["cod", "fish"]),
        ("vegetarian caesar salad in 15 min", "caesar_salad", ["vegetarian"], 15, []),
        ("vegan chickpea tahini caesar", "caesar_salad", ["vegan"], None, ["chickpea", "tahini"]),
        ("high protein grilled chicken caesar", "caesar_salad", ["high_protein"], None, ["chicken"]),
        ("gluten free tofu caesar salad", "caesar_salad", ["gluten_free"], None, ["tofu"]),
        ("gluten free dairy free chocolate cake", "chocolate_cake", ["gluten_free", "dairy_free"], None, []),
        ("quick vegan chocolate dessert under 20 minutes", "chocolate_cake", ["vegan"], 20, []),
        ("spicy cinnamon dark chocolate cake", "chocolate_cake", ["spicy"], None, ["cinnamon"]),
        ("high protein chocolate cake", "chocolate_cake", ["high_protein"], None, []),
        ("vegan black bean burger under 25 min", "hamburger", ["vegan"], 25, ["black bean"]),
        ("gluten free dairy free turkey burger", "hamburger", ["gluten_free", "dairy_free"], None, ["turkey"]),
        ("spicy jalapeno cheeseburger", "hamburger", ["spicy"], None, ["jalapeno"]),
        ("quick vegetarian mushroom burger", "hamburger", ["vegetarian"], 25, ["mushroom"]),
        ("vegetarian vegetable lasagna", "lasagna", ["vegetarian"], None, ["spinach", "zucchini"]),
        ("vegan gluten free lentil lasagna", "lasagna", ["vegan", "gluten_free"], None, ["lentil"]),
        ("quick skillet lasagna under 30 min", "lasagna", [], 30, []),
        ("high protein turkey zucchini lasagna", "lasagna", ["high_protein"], None, ["turkey", "zucchini"]),
        ("quick vegan gluten free pancakes under 20 mins", "pancakes", ["vegan", "gluten_free"], 20, []),
        ("high protein oat pancakes in 15 min", "pancakes", ["high_protein"], 15, ["oat"]),
        ("dairy free banana or apple pancakes", "pancakes", ["dairy_free"], None, []),
        ("vegan blueberry buckwheat pancakes", "pancakes", ["vegan"], None, ["blueberry", "buckwheat"]),
        # Harder synonym / cross-constraint queries (testing semantic retrieval vs exact keyword TF-IDF)
        ("meatless Italian baked pasta with spinach", "lasagna", ["vegetarian"], None, ["spinach"]),
        ("celiac friendly crispy morning breakfast stack", "pancakes", ["gluten_free"], None, []),
        ("fiery Mexican corn tortilla dinner in 20 minutes", "tacos", ["spicy"], 20, []),
        ("fast plant based Japanese broth bowl", "ramen", ["vegan"], 25, []),
        ("dairy free sweet cocoa treat", "chocolate_cake", ["dairy_free"], None, []),
        ("crunchy romaine greens with creamy garlicky tahini", "caesar_salad", ["vegan"], None, ["tahini"]),
        ("melted dark cocoa almond dessert without dairy", "chocolate_cake", ["dairy_free", "gluten_free"], None, ["almond"]),
        ("crispy corn shells with smoky poultry", "tacos", ["gluten_free"], None, ["chicken"]),
        ("flatbread with fresh mozzarella and sweet basil", "pizza", ["vegetarian"], 25, ["mozzarella", "basil"]),
        ("creamy coconut curry noodle soup without wheat", "ramen", ["gluten_free", "vegan"], 25, ["coconut"]),
        ("no meat legume burger in under 25 minutes", "hamburger", ["vegan"], 25, ["black beans", "chickpeas"]),
        ("japanese raw fish rice bowl quick meal", "sushi", ["high_protein"], 20, ["salmon", "tuna"]),
    ]

    test_queries: List[Dict[str, Any]] = []
    for q_idx, (q_text, dish_hint, req_diets, max_t, kws) in enumerate(query_specs, start=1):
        rel_ids = []
        for rec in all_recipes:
            if rec["dish_category"] != dish_hint:
                continue
            if max_t is not None and rec["prep_time_min"] > max_t:
                continue
            if req_diets and not all(d in rec["diet_tags"] for d in req_diets):
                continue
            if kws:
                blob = f"{rec['title']} {' '.join(rec['ingredients'])} {rec['description']}".lower()
                if not any(kw in blob for kw in kws):
                    continue
            rel_ids.append(rec["id"])

        if rel_ids:
            # Alternate dish_hint=None on pure semantic queries (>=41) so retrieval is tested both with and without dish hint
            hint_val = None if q_idx >= 41 else dish_hint
            test_queries.append({
                "id": f"q_{q_idx:03d}",
                "query": q_text,
                "dish_hint": hint_val,
                "relevant_ids": rel_ids,
            })

    (out_dir / "test_queries.json").write_text(json.dumps(test_queries, indent=2), encoding="utf-8")

    # Save summary metadata for citations
    hf_count = sum(1 for r in all_recipes if "Hugging Face" in r["source"])
    curated_count = len(all_recipes) - hf_count
    meta = {
        "total_recipes": len(all_recipes),
        "hf_public_recipes": hf_count,
        "curated_supplement_recipes": curated_count,
        "total_test_queries": len(test_queries),
        "classes": {c: sum(1 for r in all_recipes if r["dish_category"] == c) for c in SELECTED_CLASSES},
        "sources": ["corbt/all-recipes (Hugging Face)", "Curated domain recipe supplement"],
    }
    (out_dir / "recipes_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Built recipes & test queries:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    build_recipes_and_queries()
