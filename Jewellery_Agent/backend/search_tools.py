import chromadb
import matplotlib.pyplot as plt
import requests
import base64
import json
import torch

from sentence_transformers import SentenceTransformer
from io import BytesIO
from PIL import Image
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field, create_model
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path
from typing import List, Dict, Any, Union, Literal


class DocumentKnowledgeBase:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "guide_documents"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.colpali_model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.processor = ColPaliProcessor.from_pretrained(
            "vidore/colpali-v1.2")

    def retrieve_context_pages(self, conversation_history, k=3):
        with torch.no_grad():
            batch_query = self.processor.process_queries(
                [conversation_history]).to(self.device)
            query_embedding = self.colpali_model(**batch_query)

        multivector_query = query_embedding[0].cpu().float().numpy().tolist()

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=multivector_query,
            using="colpali",
            limit=k
        ).points

        context_images = []
        for point in search_result:
            pdf_source = point.payload["source"]
            page_num = point.payload["page_num"]

            pdf_path = f"./documents/{pdf_source}.pdf"

            images = convert_from_path(
                pdf_path, first_page=page_num, last_page=page_num)
            if images:
                context_images.append(images[0])
        return context_images


class BlueNileSearch():
    def __init__(self):
        client = chromadb.PersistentClient(path="./blue_nile_agentic_db")

        self.product_collection = client.get_collection(
            name="product_knowledge")
        self.visual_collection = client.get_collection(name="visual_index")

        self.model = SentenceTransformer("clip-ViT-B-32")

        self.document_db = DocumentKnowledgeBase()

        all_metadatas = self.product_collection.get(
            include=["metadatas"])["metadatas"]
        self.material_list = list(
            set([meta.get("material") for meta in all_metadatas]))
        self.style_list = list(set([meta.get("style")
                               for meta in all_metadatas]))
        self.gemstone_list = list(
            set([meta.get("gemstone") for meta in all_metadatas]))
        self.options_map = {
            "material": self.material_list,
            "style": self.style_list,
            "gemstone": self.gemstone_list
        }

        MaterialEnum = Literal[tuple(self.material_list + ["NO REQUIREMENT"])]
        StyleEnum = Literal[tuple(self.style_list + ["NO REQUIREMENT"])]
        GemstoneEnum = Literal[tuple(self.gemstone_list + ["NO REQUIREMENT"])]

        self.SearchIntent = create_model(
            "SearchIntent",
            reasoning=(
                str, Field(..., description="Explain your reasoning logic for every property.")),

            material=(Optional[MaterialEnum], Field(
                None, description=f"Must be one of: {self.material_list} or 'NO REQUIREMENT'."
            )),
            style=(Optional[StyleEnum], Field(
                None, description=f"Must be one of: {self.style_list} or 'NO REQUIREMENT'."
            )),
            gemstone=(Optional[GemstoneEnum], Field(
                None, description=f"Must to be one of: {self.gemstone_list} or 'NO REQUIREMENT'."
            )),

            min_price=(Union[float, str, None], Field(
                None, description="Min Price or 'NO REQUIREMENT'."
            )),
            max_price=(Union[float, str, None], Field(
                None, description="Max price or 'NO REQUIREMENT'."
            )),
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", api_key="")
        self.structed_llm = self.llm.with_structured_output(self.SearchIntent)

    def _rewrite_query(self, history, current_input):
        if not history:
            return current_input

        system_prompt = """
        You are a Search Query Refiner.
        The user is chatting with a shopping assistant.
        Your job is to combine the CHAT HISTORY and the LATEST USER INPUT
        into a single, standalone search query that can be sent to a database.
        
        Rules:
        1. Replace pronouns (it, they, that) with specific product names from history.
        2. Incorporate constraints (e.g., if history says "Diamond Ring" and user says "make it cheaper", output "Cheap Diamond Ring").
        3. Output ONLY the query string. No explanations.
        """
        conversation_str = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in history])

        input_text = f"""
        Chat History:
        {conversation_str}
        
        Latest User Input:
        {current_input}
        
        Standalone Search Query:
        """
        msg = [SystemMessage(content=system_prompt),
               HumanMessage(content=input_text)]

        response = self.llm.invoke(msg)
        return response.content.strip()

    def _get_b64_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def agentic_filtering(self, conversation_input):
        # paser input (handle list vs string)
        if isinstance(conversation_input, list):
            last_message = conversation_input[-1]
            user_text = last_message.content

            history = conversation_input[:-1]
        else:
            user_text = str(conversation_input)
            history = []

        # retrieval from knowledge base
        db_query_text = self._rewrite_query(history, user_text)
        print(f"Consulting documents for : '{db_query_text}' ...")
        retrieved_pages = self.document_db.retrieve_context_pages(
            db_query_text)

        extraction_system_msg = SystemMessage(content="""                
        You are an Expert Jewelry Consultant. Your goal is to translate user requests into precise search filters.

        ### CORE REASONING STRATEGY (HYBRID APPROACH):
        You have two sources of intelligence. Use them in this order:
        1. **Context Documents:** Check the provided images/text for specific brand definitions or collections that might override general knowledge.
        2. **Internal Expert Knowledge:** Use your training on jewelry terms (e.g., you know "sparkling" usually implies 'Halo', 'Pavé', or 'Radiant Cut').
    
        ### RULES FOR AGGRESSIVE INFERENCE:

        1. **INTERPRET VAGUE TERMS (Use World Knowledge):**
        - **"Sparkling/Bling":** If the docs are silent, use your knowledge. Map to `style='Halo'` or `style='Pavé'`.
        - **"Vintage/Old School":** Map to `style='Vintage'` or `style='Milgrain'`.
        - **"Modern/Sleek":** Map to `style='Bezel'` or `style='Solitaire'`.
        - **Logic:** Do not return None. Make an educated guess based on standard industry terminology.

        2. **PRICE LOGIC (Market Estimation):**
        - **If the documents show prices:** Use those as hard anchors.
        - **If documents DO NOT show prices:** Use general market estimation for fine jewelry.
            - "Cheap/Affordable Ring"
            - "Mid-range"
            - "Luxury/Expensive"
        - **Reasoning the min price and the max price at the same time.
        - **Explicitly state your logic:** "The documents didn't list prices, so I am using standard market definition for 'cheap diamond ring' (<$1500)."

        3. **HANDLING LISTS (Strict Enum Mapping):**
        - You must map your inferred concept to the closest match in the provided `material` or `style` lists.
        - If you infer "Rose Gold" but the list only has "14k Rose Gold", select "14k Rose Gold".

        4. **"NO REQUIREMENT" vs None:**
        - Use "NO REQUIREMENT" *only* if the user explicitly says "I don't care" or "Any".
        - Use `None` if the user didn't mention it and you cannot safely infer it.
        """)

        content_parts: List[Union[str, Dict]] = [
            {"type": "text",
                "text": f"User Request: {user_text}\n\n[SYSTEM INJECTION] I have retrieved the following pages for you to check before answering:"}
        ]

        for page_img in retrieved_pages:
            b64_str = self._get_b64_image(page_img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })

        current_multimodal_message = HumanMessage(content=content_parts)
        messages_to_send = [extraction_system_msg] + \
            history + [current_multimodal_message]

        # execution
        intent = self.structed_llm.invoke(messages_to_send)
        print("reasoning: ", intent.reasoning)
        print(f"intent: {intent}")

        # check if all attributes are filled
        for field_name, value in intent.model_dump().items():
            if value is None:
                suggest_list = self.options_map[field_name]
                return f"CLARIFICATION_NEEDED: The user wants a JEWELRY/DIAMOND gift, but the query was too vague. \
                        Acting as an Expert Jeweler, ask them to specify their preferences for {field_name}, which should be in {suggest_list}."

        active_filters = {}
        for field_name, value in intent.model_dump().items():
            if (value == "NO REQUIREMENT" or "price" in field_name or "reasoning" in field_name):
                continue
            else:
                active_filters[field_name] = {field_name: value}

        price_conditions = []
        if (intent.min_price is not None and intent.min_price != "NO REQUIREMENT"):
            price_conditions.append({"price": {"$gte": intent.min_price}})
        if (intent.max_price is not None and intent.max_price != "NO REQUIREMENT"):
            price_conditions.append({"price": {"$lte": intent.max_price}})

        if price_conditions:
            if len(price_conditions) > 1:
                active_filters["price"] = {"$and": price_conditions}
            else:
                active_filters["price"] = price_conditions[0]

        print("active filters = {}".format(active_filters))

        all_rules = list(active_filters.values())
        strict_where = {"$and": all_rules} if len(
            all_rules) > 1 else all_rules[0]

        results = self.product_collection.get(
            where=strict_where, include=["metadatas"])
        valid_ids = results["ids"]

        if not valid_ids:
            diagnosis_lines = []

            for filter_name in active_filters.keys():
                relaxed_rules = [
                    rule for k, rule in active_filters.items() if k != filter_name
                ]

                if not relaxed_rules:
                    count = self.product_collection.count()
                else:
                    relaxed_where = {"$and": relaxed_rules} if len(
                        relaxed_rules) > 1 else relaxed_rules[0]
                    count = len(self.product_collection.get(
                        where=relaxed_where, include=["metadatas"]))

                if count > 0:
                    diagnosis_lines.append(
                        f"-Option: Keep everything but change '{filter_name}', we have {count} items.")
            suggestion = "\n".join(diagnosis_lines)

            return_str = f"""
            SYSTEM_NOTICE: No exact matches found for the user's query.
            
            YOUR TASK:
            You are a helpful Shopping Assistant. Do NOT say "I found zero results" in a robotic way.
            Instead, apologize gently, and guide the user to adjust their search based on the diagnostic data below.

            DIAGNOSTIC DATA:
            {suggestion}

            EXAMPLE RESPONSE:
            "I couldn't find a ring that matches ALL your criteria (Rose Gold, Halo, under $500). 
            However, if you are flexible with the price, I have 12 Rose Gold Halo rings available. 
            Or, if you stick to the $500 budget, I have 5 Solitaire rings available. Which would you prefer?"
            """
            print(f"return str: {return_str}")
            return return_str
        else:
            return valid_ids

    def agentic_search(self, user_query, valid_ids, top_k=5):
        query_vector = self.model.encode(user_query).tolist()

        fetch_k = top_k * 4

        search_args = {
            "query_embeddings": [query_vector],
            "n_results": fetch_k,
            "include": ["metadatas", "distances"]
        }

        if valid_ids:
            search_args["where"] = {"parent_id": {"$in": valid_ids}}

        visual_results = self.visual_collection.query(**search_args)

        final_results = []
        seen_products = set()

        if visual_results["metadatas"] and visual_results["metadatas"][0]:
            metas = visual_results["metadatas"][0]
            dists = visual_results["distances"][0]

            for i, meta in enumerate(metas):
                p_id = meta["parent_id"]

                if p_id in seen_products:
                    continue

                seen_products.add(p_id)

                final_results.append({
                    "score": dists[i],
                    # "url": meta["url"],
                    "image_url": meta["image_url"],
                    "name": meta["name"],
                    "price": meta["price"],
                    "product_id": p_id,
                    "view_index": meta.get("view_index", "unknown")
                })

                if len(final_results) >= top_k:
                    break

        return final_results

    def visualize_results(self, results):
        fig, axes = plt.subplots(2, 3, figsize=(25, 25))
        axes = axes.flatten()

        for i, result in enumerate(results):
            img_response = requests.get(
                result["image_url"], stream=True, timeout=20)
            img = Image.open(BytesIO(img_response.content))

            axes[i].imshow(img)
            axes[i].set_title(f"{result['name']}\n${result['price']}")
            axes[i].axis("off")

        for j in range(len(results), len(axes)):
            axes[j].axis("off")

        plt.savefig("results.png")
        plt.close()

    def format_result(self, results):
        output_str = "Here are the best matches I found:\n\n"

        if not results:
            return "I couldn't find any items matching that description"

        for r in results:
            output_str += f"### {r.get('name', 'Unknown Item')} - ${r.get('price', '???')}\n"

            output_str += "\n---\n"

        return output_str

    def generate_response(self, conversation_input, results):
        image_gallery = []
        lean_results = []

        for index, item in enumerate(results):
            image_url = item["image_url"]
            image_response = requests.get(image_url, stream=True, timeout=20)
            content_type = image_response.headers.get(
                "Content-Type", "image/jpg")
            encoded_image = base64.b64encode(
                image_response.content).decode("utf-8")
            image_gallery.append(f"data:{content_type};base64,{encoded_image}")

            lean_results.append({
                "index": index,
                "name": item.get("name"),
                "price": item.get("price"),
                "description": item.get("description", "No description")
            })

        context_str = json.dumps(lean_results, indent=4)

        system_prompt = f"""
        You are a helpful Jewellery Assistant.
        The user searched for: "{conversation_input}"
        
        Here is the data found in the database:
        {context_str}
        
        INSTRUCTIONS:
        1. Write a natural, engaging response recommending these items.
        2. Do not just list them; compare them or highlight features.
        3. When you mention an item, you MUST display its image using Markdown format: ![Item Name](image_INDEX).
        4. The 'INDEX' corresponds to the 'index' field in the data above.
        
        Example format:
        "First, I found this amazing Emerald Ring for $500. It features a gold band:
        ![Emerald Ring](image_0)
        
        If you prefer something cheaper, check out this Silver Band..."
        """
        llm_output = self.llm.invoke(system_prompt).content

        return json.dumps({
            "response": llm_output,
            "images": image_gallery
        })
