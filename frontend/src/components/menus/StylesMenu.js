import React, { useRef, useState, useEffect } from "react";
import {
  Undo2,
  Redo2,
  Search,
  SplitSquareHorizontal,
  Sparkles,
  Loader2,
} from "lucide-react";
import { IconUndo, IconRedo, IconSearchFooter, IconSplit, IconClose, IconCheck } from '../../constants';
import { loadStyleTransferModel, runStyleTransfer, unloadStyleTransferModel } from '../../api/styleTransferAPI';
import { toast } from 'sonner';

const StylesMenu = ({ onBack, imageSrc, onImageUpdate }) => {
  const scrollContainerRef = useRef(null);
  const containerRef = useRef(null);
  const [selectedStyleIndex, setSelectedStyleIndex] = useState(0);
  const [showGenAI, setShowGenAI] = useState(false);
  const [genAIPrompt, setGenAIPrompt] = useState("");
  const [showStylesText, setShowStylesText] = useState(true);
  const [isClosing, setIsClosing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  // --- INTEGRATED LORA STYLES DATA ---
  const loras = [
    {
        "name": "Greenery - SD 1.5 and SDXL",
        "description":"A Stable Diffusion 1.5 LoRA that transforms ordinary or urban scenes into lush, green natural environments. It adds dense foliage, moss, plants, and natural textures to images, making them appear overgrown or reclaimed by nature. The model enhances lighting with soft cinematic tones and vivid greenery while maintaining realistic detail and depth. Ideal for generating forests, jungles, natural landscapes, or turning man-made environments into eco-rich, overgrown sceneries. Works best with moderate strength to preserve human subjects while modifying the background.",
        "path_to_lora":"S/Greenery.safetensors",
        "strength":0.9,
        "guidance":7.0,
        "num_inference_steps":200,
        "image_description": "A man standing in an urbanic setting in front of a lake",
        "positive_prompt":"Create a scenic natural environemnt, lush green natural background, nature reclaiming civilization, vivid greenery, ultra realistic, highly detailed, <lora:Greenery:0.4>",
        "negative_prompt":"tie, distorted human, erased person, missing face, blur, low quality, text, watermark, logo, oversaturated, artifacts, deformed structures, unrealistic textures, unnatural patterns, distorted perspective",
        "trigger_words": ["greenery"]
    },
    {
        "name": "Oil painting(oil brush stroke) ",
        "description":"Used for generating oil panting based images",
        "path_to_lora":"/root/workspace/S/sd1.5loras/bichu-v0612.safetensors",
        "strength":0.9,
        "guidance":7.0,
        "num_inference_steps":200,
        "image_description": "A man standing in an urbanic setting in front of a lake",
        "positive_prompt":"portrait of a person, oil painting style, artistic brush strokes, rich texture, soft lighting, detailed face, warm tones, painterly realism, canvas texture, dramatic contrast, Rembrandt-style lighting, masterpiece, highly detailed, bichu,<lora:Oil painting(oil brush stroke) - 油画笔触:0.5>",
        "negative_prompt":"photorealistic, modern digital render, blur, distorted face, extra limbs, cartoon, low contrast, flat color, deformed anatomy, text, watermark, logo, noisy background, pixelated texture, unnatural proportions",
        "trigger_words": ["bichu"]
    },
    {
        "name": "xiaorenshu",
        "description":"Used for generating japanese style images",
        "path_to_lora":"/root/workspace/S/sd1.5loras/xrs2.0.safetensors",
        "strength":0.9,
        "guidance":7.0,
        "num_inference_steps":200,
        "image_description": "A man standing in an urbanic setting in front of a lake",
        "positive_prompt":"Convert to japanese art style <lora:小人书·连环画 xiaorenshu:0.4>",
        "negative_prompt":"blurry, distorted face, low detail, overexposed, flat lighting, low resolution, multiple faces, deformed anatomy,oversaturated colors, harsh shadows, watermark, text, unnatural pose, unrealistic proportions, artifacts, noise",
        "trigger_words": [""]
    },
    {
        "name": "cosmic_nebula",
        "description":"Utransforms images into vibrant cosmic scenes filled with swirling nebula clouds, glowing stardust, and ethereal interstellar lightin",
        "path_to_lora":"/root/workspace/S/sd1.5loras/cosmic.safetensors",
        "strength":0.9,
        "guidance":0.7,
        "num_inference_steps":70,
        "image_description": "A young chinese boy laughing joyfully in a checked shirt",
        "positive_prompt":"martius_nebula, a young chinese boy laughing joyfully in checked shirt, cosmic nebula clouds swirling in the sky behind him, glowing purple and blue stardust drifting around, soft astral light reflecting on his face, vibrant interstellar colors surrounding him, dreamy cosmic atmosphere, ethereal lighting, high detail, realistic anatomy, cosmic energy glow outlining his silhouette",
        "negative_prompt":"low detail, blurry, dull colors, washed out, flat lighting, cartoonish, distorted face, deformed hands, extra limbs, mutated features, bad anatomy, watermark, text, logo, grainy, noisy background, overexposed, underexposed",
        "trigger_words": ["martius_nebula"]
    },
    {
        "name": "steel_magic",
        "description":"adds molten-metal runes, forged-steel energy, and dramatic magical armor effects to any character.",
        "path_to_lora":"/root/workspace/S/sd1.5loras/steel.safetensors",
        "strength":0.65,
        "guidance":8.5,
        "num_inference_steps":70,
        "image_description": "a black man standing in gray shirt and black jeans",
        "positive_prompt":["DonMSt33lM4g1c, a Black man standing confidently, short well-trimmed hair, realistic face and anatomy",
            "arcane steel-magic energy swirling subtly around him",
            "glowing molten-metal runes orbiting his body like forged symbols",
            "faint sparks and metallic dust drifting through the air",
            "cool blue and fiery orange steel-forge lighting reflecting on his clothes",
            "epic steel-magic fantasy atmosphere, sharp details, high clarity"],
        "negative_prompt": ["fiery face, blurry, low detail, cartoonish",
            "distorted hands, extra limbs, bad anatomy",
            "washed out colors, text, watermark",
            "flat lighting, unrealistic proportions"],  
        "trigger_words": ["mDonMSt33lM4g1c"]
    },
    {
        "name": "retro_game_art",
        "description":"transforms images into vibrant 16-bit retro game art with bold outlines, pixel shading, and classic arcade aesthetics",
        "path_to_lora":"/root/workspace/S/sd1.5loras/retro.safetensors",
        "strength":0.8,
        "guidance":7,
        "num_inference_steps":70,
        "image_description": "a black man standing in gray shirt and black jeans",
        "positive_prompt":["r3tr0, a Black man standing confidently in a retro 16-bit game art style",
            "chunky pixel-art shading, bold outlines, vibrant retro color palette",
            "dynamic side-scroll hero stance, slight forward lean, powerful silhouette",
            "retro game background with neon-lit city buildings and pixel clouds",
            "crisp pixel details, sharp sprites, authentic SNES/Genesis game aesthetic"],
        "negative_prompt" : "blurry pixels, low detail, broken sprite shapes, pastel colors, text, watermark, realistic photo style",
        "trigger_words": ["r3tr0"]
    },
    {
        "name": "ghibli",
        "description":"LoRA for Stable Diffusion 1.5 that transfers the signature Studio Ghibli art style—soft colors, hand-painted textures, and whimsical lighting—onto any image or concept. Ideal for Ghibli-style landscapes, characters, and illustrations",
        "path_to_lora":"styleLorasSD1.5/ghibli.safetensors",
        "strength":0.9,
        "guidance":5.5,
        "num_inference_steps":60,
        "image_description": "elephant in a grassland with hills in background and cloudy sky",
        "positive_prompt": [
        "StdGBRedmAF , elephant , detailed hand-painted illustration, dreamy atmosphere, ",
        "soft watercolor shading, warm and nostalgic lighting, lush scenery, ",
        "cinematic composition, whimsical and heartwarming tone, ",
        "painterly brush strokes, vibrant natural colors"
        ],
        "negative_prompt": [
        "photorealistic, realistic, CGI, 3D render, plastic texture, harsh lighting, ",
        "text, watermark, signature, oversaturated, distorted"
        ],
        "trigger_words": ["Studio Ghibli", "StdGBRedmAF"]
    },
    {
        "name": "nyalia",
        "description":"A versatile LoRA for Stable Diffusion that applies the distinct “Nyalia” AI art aesthetic: ethereal lighting, soft gradients, and refined character focus. Ideal for portraits, fantasy scenes, and stylised concept art.",
        "path_to_lora":"styleLorasSD1.5/nyalia.safetensors",
        "strength":0.9,
        "guidance":7.0,
        "num_inference_steps":200,
        "image_description": "young smiling black boy with curly hair in a blue tshirt.",
        "positive_prompt":"nyalia A portrait of a young Black boy with radiant brown skin and deep expressive eyes, softly illuminated by golden light, wearing a simple white shirt, surrounded by dreamy pastel tones and glowing particles, in the Nyalia AI style — painterly textures, cinematic color grading, elegant composition, ultra-realistic yet artistic, masterpiece quality.",
        "negative_prompt":"blurry, bad anatomy, deformed, extra limbs, glitch, text, watermark, signature, cartoonish, oversaturated neon, flat lighting.",
        "trigger_words": ["nyalia"]
    },
    {
        "name": "gta",
        "description":"A versatile LoRA for Stable Diffusion that applies the distinct “Grand Theft Auto” art aesthetic: gritty urban environments, dramatic lighting, and stylized character designs. Ideal for action scenes, character portraits, and dynamic compositions.",
        "path_to_lora":"styleLorasSD1.5/gta.safetensors",
        "strength":0.9,
        "guidance":7.0,
        "num_inference_steps":200,
        "image_description": "young smiling black boy with curly hair in a blue tshirt.",
        "positive_prompt":"gr4nd_th3ft4_4 A portrait of a young Black boy  standing in an urban city street at sunset, wearing a leather jacket and jeans, confident expression, hands in pockets, dramatic cinematic lighting, detailed face, stylized shading and contour lines, painted in Grand Theft Auto IV loading screen art style, semi-realistic comic illustration, gritty yet vibrant tone, detailed background, masterpiece, 4K",
        "negative_prompt":"blurry, bad anatomy, deformed, extra limbs, glitch, text, watermark, signature, cartoonish, oversaturated neon, flat lighting.",
        "trigger_words": ["gr4nd_th3ft4_4"]
    },
    {
        "name": "zombie",
        "description":"A versatile LoRA for Stable Diffusion that applies the distinct “Zombie” art aesthetic: post-apocalyptic environments, dramatic lighting, and stylized character designs. Ideal for horror scenes, character portraits, and dynamic compositions.",
        "path_to_lora":"styleLorasSD1.5/zombie.safetensors",
        "strength":0.9,
        "guidance":6.0,
        "num_inference_steps":200,
        "image_description": "young smiling black boy with curly hair in a blue tshirt.",
        "positive_prompt":"post-apocalyptic world post-apocalyptic worldClose-up portrait of a young Black boy, face half turned toward camera, wide fearful eyes, dark skin with undead pallor beginning, subtle decay details: cracked dry skin, faint greenish tint under cheeks, faintly visible bone structure under skin, tattered school hoodie, ambient moonlight from above, cinematic horror lighting, dripping slight mist in background, in the style of Uzombies SD15 — ultra-detailed zombie concept art, high resolution, textured skin, dramatic shadows.",
        "negative_prompt":"blurry, bad anatomy, deformed, extra limbs, glitch, text, watermark, signature, cartoonish, oversaturated neon, flat lighting.",
        "trigger_words": ["post-apocalyptic world", "zombie","zombies"]
    },
    {
        "name": "toy_story_pixar",
        "path_to_lora":"styleLorasSD1.5/disney.safetensors",
        "strength":0.75,
        "guidance":7.1,
        "num_inference_steps":100,
        "image_description": "young smiling black boy with curly hair in a blue tshirt",
        "positive_prompt": [
            "JessieWaifu, Toy Story style Pixar 3D render, young Black boy toy character, wearing a cowboy hat, ",
            "colorful shirt, animal print pants, boots, and belt, expressive face and bright smile, ",
            "soft cinematic lighting, detailed fabric and plastic textures, high-quality Pixar 3D render, ",
            "colorful bedroom toy environment, warm and whimsical atmosphere, ",
            "Disney Pixar movie aesthetic, vibrant color palette"
        ],
        "negative_prompt": [
            "realistic photo, creepy, low-poly, distorted face, text, watermark, ",
            "oversaturated, harsh lighting, messy background, extra limbs, broken anatomy"
        ],
        "trigger_words": ["JessieWaifu", "(hat, shirt, animal print pants, boots, belt)", "single braid"]
    },
    {
        "name": "cyberpunkAI",
        "path_to_lora":"styleLorasSD1.5/cyberpunk.safetensors",
        "strength":0.75,
        "guidance":6.5,
        "num_inference_steps":100,
        "image_description": "young smiling black boy with curly hair in a blue tshirt",
        "positive_prompt": [
            "CyberpunkAI style, neon-lit futuristic cityscape, young Black boy in high-tech streetwear", 
            "glowing cybernetic visor, chrome jacket with neon trims, reflective wet pavement",
            "dark alley illuminated by holograms, cinematic lighting and deep shadows",
            "detailed 3D render, vivid neon palette, high-tech urban atmosphere, animated movie quality"
        ],
        "negative_prompt": [
            "blurry, low resolution, photorealistic, grainy, text, watermark, distorted anatomy" ,
            "pastel colors, daylight scene, low contrast"
        ],
        "trigger_words": ["CyberpunkAI"]
    },
    {
        "name": "sketch",
        "path_to_lora":"styleLorasSD1.5/sketch.safetensors",
        "strength":0.99,
        "guidance":6.8,
        "num_inference_steps":100,
        "image_description": "young smiling black boy with curly hair in a blue tshirt",
        "positive_prompt": [
            "black and white pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch, ",
            "no colors, traditional hand-drawn look,",
            "unfinished, hatching texture, ",
            "anime style illustration of a young Black boy, ",
            "large expressive eyes, slightly tousled hair, smiley face",
            "focus on line work and texture rather than full color",
            "dynamic pose, slight motion in his hair and clothes, ",
            "soft ambient light, subtle shadowing "
        ],
        "negative_prompt": [
            "photorealistic, full color, smooth lines, CG render, ",
            "watermark, text, signature, oversaturated, ",
            "blurred, low detail, distorted anatomy, extra limbs"
        ],
        "trigger_words": [
            "(Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch)",
            "(unfinished, hatching (texture))"
        ]
    },
    {
        "name": "brush_painting",
        "path_to_lora":"styleLorasSD1.5/painting.safetensors",
        "strength":0.8,
        "guidance":8,
        "num_inference_steps":200,
        "image_description": "elephant in a grassland with hills in background and cloudy sky",
        "positive_prompt": [
            "BigBrush painting artstyle, impressionist oil painting, bold brush strokes, ",
            "elephant in a wide grassland with rolling hills in the background, ",
            "dramatic cloudy sky with expressive strokes, soft sunlight breaking through clouds, ",
            "rich textured canvas feel, painterly composition, natural earth tones blended with vivid highlights, ",
            "hand-painted aesthetic, cinematic depth, artistic mood, ",
            "thick paint texture and visible brushwork throughout"
        ],
        "negative_prompt": [
            "photorealistic, smooth digital render, flat shading, CGI, ",
            "sharp edges, 3D model look, text, watermark, oversaturated colors, ",
            "cartoonish style, distorted anatomy, low detail"
        ],
        "trigger_words": []
    }
];

  const styles = loras.map((lora, index) => {
    let creativeName = lora.name;
    let imageUrl = "";

    switch (lora.name) {
        case "Greenery - SD 1.5 and SDXL":
            creativeName = "Nature Reclaims";
            imageUrl = require("../../assets/nature_reclaims.jpg");
            break;
        case "Oil painting(oil brush stroke) ":
            creativeName = "Masterpiece Oil";
            imageUrl = require("../../assets/masterpiece_oil.jpg");
            break;
        case "xiaorenshu":
            creativeName = "Japanese Comic Art";
            imageUrl = require("../../assets/japanese_comic_art.jpg");
            break;
        case "cosmic_nebula":
            creativeName = "Cosmic Nebula";
            imageUrl = require("../../assets/cosmic_nebula.jpg");
            break;
        case "steel_magic":
            creativeName = "Arcane Steel Magic";
            imageUrl = require("../../assets/arcane_steel_magic.jpg");
            break;
        case "retro_game_art":
            creativeName = "16-Bit Arcade";
            imageUrl = require("../../assets/16-bit_arcade.jpg");
            break;
        case "ghibli":
            creativeName = "Studio Ghibli Film";
            imageUrl = require("../../assets/studio_ghibli_film.jpg");
            break;
        case "nyalia":
            creativeName = "Ethereal Portrait";
            imageUrl = require("../../assets/ethereal_portrait.jpeg");
            break;
        case "gta":
            creativeName = "Grand Theft Auto";
            imageUrl = require("../../assets/grand_theft_auto.png");
            break;
        case "zombie":
            creativeName = "Post-Apocalyptic Zombie";
            imageUrl = require("../../assets/post-apocalyptic_zombie.jpeg");
            break;
        case "toy_story_pixar":
            creativeName = "Pixar 3D Toy";
            imageUrl = require("../../assets/pixar_3d_toy.jpg");
            break;
        case "cyberpunkAI":
            creativeName = "Cyberpunk Neon";
            imageUrl = require("../../assets/cyberpunk_neon.jpg");
            break;
        case "sketch":
            creativeName = "Pencil Sketch";
            imageUrl = require("../../assets/pencil_sketch.jpg");
            break;
        case "brush_painting":
            creativeName = "Impressionist Brush";
            imageUrl = require("../../assets/impressionist_brush.jpg");
            break;
        default:
            creativeName = lora.name;
            imageUrl = require("../../assets/default_placeholder.jpeg");
    }
    
    return {
        id: index + 1,
        name: creativeName,
        image: imageUrl,
        lora: lora, // Keep the full lora object for backend use
    };
  });
  // --- END OF INTEGRATED LORA STYLES DATA ---


  // Handle scroll and auto-select item
  const handleScroll = () => {
    if (!scrollContainerRef.current) return;

    const container = scrollContainerRef.current;
    const scrollLeft = container.scrollLeft;
    // Card width (72px) + gap (16px) = 88px
    const cardWidth = 88; 
    const totalWidth = cardWidth * styles.length;

    // Get the actual index within the original styles array
    const index = Math.round(scrollLeft / cardWidth) % styles.length;
    setSelectedStyleIndex(index);

    // Reset scroll position when reaching the end or beginning for true infinite loop
    if (scrollLeft >= totalWidth * 3.5) {
      container.scrollLeft = totalWidth * 1.5;
    } else if (scrollLeft <= totalWidth * 0.5 && scrollLeft > 0) {
      container.scrollLeft = totalWidth * 2.5;
    }
  };

  // Scroll to specific style
  const scrollToStyle = (index) => {
    if (!scrollContainerRef.current) return;

    const container = scrollContainerRef.current;
    const cardWidth = 88;
    // Calculate the position to scroll to for the centered item (in the middle set of copies)
    // The styles array is copied 5 times. The middle copy is at index 2.
    const middleStartIndex = styles.length * 2;
    const scrollPosition = (middleStartIndex + index) * cardWidth;

    container.scrollTo({
      left: scrollPosition,
      behavior: "smooth",
    });
  };

  useEffect(() => {
    // Initialize scroll position to middle set for infinite loop (start at styles.length * 88 * 2)
    if (scrollContainerRef.current && styles.length > 0) {
      const cardWidth = 88;
      // Scroll to the first item of the third copy (index 2)
      scrollContainerRef.current.scrollLeft = cardWidth * styles.length * 2;
    }
  }, [styles.length]);

  useEffect(() => {
    // Show "Styles" text for 2 seconds, then switch to selected style name
    setShowStylesText(true);
    const timer = setTimeout(() => {
      setShowStylesText(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // When selected style changes, briefly show the new style name
    if (!showStylesText) {
      // Style name is already showing, just let it update
    }
  }, [selectedStyleIndex]);

  useEffect(() => {
    if (!isClosing) return;
    const el = containerRef.current;
    if (!el) return;

    const onAnimationEnd = (e) => {
      // only react to the container's animation end
      if (e.target !== el) return;
      if (typeof onBack === "function") onBack();
    };

    el.addEventListener("animationend", onAnimationEnd);
    return () => el.removeEventListener("animationend", onAnimationEnd);
  }, [isClosing, onBack]);

  // Load the style transfer model when GenAI panel opens
  useEffect(() => {
    if (showGenAI && !isModelLoaded) {
      loadStyleTransferModel()
        .then(() => {
          setIsModelLoaded(true);
          toast.success('Style transfer model ready');
        })
        .catch((error) => {
          console.error('Failed to load model:', error);
          toast.error('Failed to load style transfer model');
        });
    }
  }, [showGenAI, isModelLoaded]);

  // Handle style transfer generation
  const handleGenerate = async () => {
    if (!genAIPrompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    if (!imageSrc) {
      toast.error('Please upload an image first');
      return;
    }

    setIsGenerating(true);

    try {
      const result = await runStyleTransfer(genAIPrompt, imageSrc);

      // Unload model after generation to free up resources
      unloadStyleTransferModel().catch(err => console.warn('Failed to unload style transfer model:', err));

      let imageUrl = null;

      if (result.blob) {
        // Response was a blob
        imageUrl = URL.createObjectURL(result.blob);
      } else if (result.image_base64 || result.image_b64 || result.image) {
        // Response was base64
        const b64 = result.image_base64 || result.image_b64 || result.image;
        const fetchSrc = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
        imageUrl = fetchSrc;
      } else if (result.image_url) {
        // Response was a URL
        imageUrl = result.image_url;
      }

      if (imageUrl && onImageUpdate) {
        onImageUpdate(imageUrl);
        toast.success('Style transfer complete!');
        // Close the menu after successful generation
        setIsClosing(true);
      } else {
        toast.error('No image returned from style transfer');
      }
    } catch (error) {
      console.error('Style transfer failed:', error);
      toast.error('Style transfer failed: ' + (error.message || 'Unknown error'));
      // Also unload on error
      unloadStyleTransferModel().catch(err => console.warn('Failed to unload style transfer model:', err));
    } finally {
      setIsGenerating(false);
      setIsModelLoaded(false); // Reset so it reloads next time
    }
  };

  return (
    <div
      ref={containerRef}
      className={`fixed bottom-0 left-1/2 -translate-x-1/2 z-40 overflow-hidden ${
        isClosing ? "animate-collapseStyles" : "animate-expandStyles"
      }`}
    >
      {/* Toolbar with dynamic text and icons */}
      <div className={`flex items-center justify-between px-4 py-3 bg-black/90 backdrop-blur-sm ${
        isClosing ? "animate-collapseContent" : ""
      }`}>
        <div className="flex items-center gap-3">
          {showGenAI ? (
            <button
              onClick={() => {
                setShowGenAI(false);
                setGenAIPrompt("");
              }}
              className="text-white hover:text-[#FF4458] transition-colors"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15 19l-7-7 7-7"
                />
              </svg>
            </button>
          ) : (
            <>
              <button className="text-white/80 hover:text-white transition-colors">
                <IconUndo size={20} />
              </button>
              <button className="text-white/80 hover:text-white transition-colors">
                <IconRedo size={20} />
              </button>
            </>
          )}
        </div>

        <h2 className="text-white text-sm font-medium transition-opacity duration-300">
          {showGenAI
            ? "Generate"
            : showStylesText
            ? "Styles"
            : styles[selectedStyleIndex]?.name || "Styles"}
        </h2>

        <div className="flex items-center gap-3">
          {!showGenAI && (
            <>
              <button className="text-white/80 hover:text-white transition-colors">
                <IconSearchFooter size={20} />
              </button>
              <button className="text-white/80 hover:text-white transition-colors">
                <IconSplit size={20} />
              </button>
            </>
          )}
          <button
            onClick={() => {
              if (isClosing) return;
              setIsClosing(true);
            }}
            className="text-white/80 hover:text-[#FF4458] transition-colors"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Styles Carousel or Gen AI Overlay */}
      {!showGenAI ? (
        <div className={`bg-[#1a1a1a] px-4 py-6 rounded-t-3xl relative ${
          isClosing ? "animate-collapseContent" : ""
        }`}>
          <div
            ref={scrollContainerRef}
            onScroll={handleScroll}
            className="flex gap-4 overflow-x-auto scrollbar-hide snap-x snap-mandatory pb-2"
            style={{ scrollSnapType: "x mandatory", paddingRight: "250px" }}
          >
            {/* Render styles 5 times for seamless infinite loop */}
            {[...Array(5)].map((_, setIndex) =>
              styles.map((style, index) => (
                <div
                  key={`${setIndex}-${style.id}`}
                  onClick={() => scrollToStyle(index)} // Scroll to the original index (middle set)
                  className="flex-shrink-0 snap-start"
                >
                  <div
                    className={`
                      w-[72px] h-[96px] rounded-xl overflow-hidden
                      border-2 transition-all duration-200 cursor-pointer
                      ${
                        selectedStyleIndex === index
                          ? "border-[#FF4458]"
                          : "border-transparent opacity-50"
                      }
                    `}
                  >
                    <img
                      src={style.image}
                      alt={style.name}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Gen AI Button - Fixed overlay on right */}
          <div className="absolute right-4 bottom-4 flex items-center pointer-events-none">
            <div className="pointer-events-auto">
              <button
                onClick={() => setShowGenAI(true)}
                className="w-12 h-12 rounded-full bg-[#FF4458] hover:bg-[#FF5568] transition-all flex items-center justify-center shadow-lg"
              >
                <Sparkles size={24} className="text-white" />
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* Gen AI Prompt Panel - Replaces carousel */
        <div className={`bg-[#1a1a1a] px-6 py-7 rounded-t-3xl animate-fadeIn ${
          isClosing ? "animate-collapseContent" : ""
        }`}>
          <div className="relative">
            <textarea
              value={genAIPrompt}
              onChange={(e) => setGenAIPrompt(e.target.value)}
              placeholder="A futuristic cityscape in neon colors..."
              className="w-full bg-[#2a2a2a] text-white placeholder:text-white/40 rounded-3xl px-5 py-4 pr-16 resize-none focus:outline-none focus:ring-2 focus:ring-[#FF4458]/50 text-[15px]"
              rows={3}
              disabled={isGenerating}
            />
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !genAIPrompt.trim()}
              className={`absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full transition-all flex items-center justify-center shadow-lg ${
                isGenerating || !genAIPrompt.trim()
                  ? 'bg-[#FF4458]/50 cursor-not-allowed'
                  : 'bg-[#FF4458] hover:bg-[#FF5568]'
              }`}
            >
              {isGenerating ? (
                <Loader2 size={20} className="text-white animate-spin" />
              ) : (
                <svg
                  className="w-5 h-5 text-white"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5 10l7-7m0 0l7 7m-7-7v18"
                  />
                </svg>
              )}
            </button>
          </div>
          {isGenerating && (
            <p className="text-white/60 text-sm text-center mt-3">
              Generating style transfer...
            </p>
          )}
        </div>
      )}

      <style jsx>{`
        @keyframes expand-from-circle-bottom {
          0% {
            width: 4rem;
            height: 4rem;
          }
          100% {
            width: 100vw;
            height: auto;
          }
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        .animate-expandStyles {
          animation: expand-from-circle-bottom 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }

        .animate-collapseStyles {
          animation: collapse-to-circle-bottom 0.4s cubic-bezier(0.64, 0, 0.78, 0) forwards;
        }

        @keyframes collapse-to-circle-bottom {
          0% {
            width: 100vw;
            height: auto;
          }
          100% {
            width: 4rem;
            height: 4rem;
          }
        }

        @keyframes collapse-content {
          0% {
            width: 100%;
            opacity: 1;
            transform: translateX(0) scaleX(1);
          }
          70% {
            opacity: 1;
            transform: translateX(0) scaleX(1);
          }
          100% {
            width: 100%;
            opacity: 0;
            transform: translateX(calc(50% - 50%)) scaleX(0);
          }
        }

        .animate-collapseContent {
          animation: collapse-content 0.55s cubic-bezier(0.64, 0, 0.78, 0) forwards;
          transform-origin: center;
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }

        @media (min-width: 640px) {
          @keyframes expand-from-circle-bottom {
            0% {
              width: 5rem;
              height: 5rem;
            }
            100% {
              width: 100vw;
              height: auto;
            }
          }
        }
      `}</style>
    </div>
  );
};

export default StylesMenu;