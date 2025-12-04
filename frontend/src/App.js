import React, { useState, useEffect, useCallback } from "react";
import canvasicon from "./icons/Vector.svg";
import discovericon from "./icons/purple.svg";
import photoroomicon from "./icons/blue.svg";
import {useNavigate, Link, BrowserRouter, Routes, Route } from "react-router-dom";
import {Lightbulb} from 'lucide-react';
import { RippleEffect, useCardClickAnimation } from "./PageTransition";

/**
 * Mobile-only responsive App
 * - Ensures everything fits within 100vh on mobile (no vertical scroll)
 * - Keeps original card heights: mobile 200px, desktop 220px
 * - Disables vertical overscroll/bounce and pinch zoom prevention (mobile only)
 * - Allows horizontal swipe (touch-action: pan-x)
 */

export default function App() {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia("(max-width: 768px)").matches : false
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(max-width: 768px)");
    const handle = (ev) => setIsMobile(ev.matches);
    mq.addEventListener ? mq.addEventListener("change", handle) : mq.addListener(handle);
    return () => (mq.removeEventListener ? mq.removeEventListener("change", handle) : mq.removeListener(handle));
  }, []);

  // Prevent pinch and gesturestart (mobile) + prevent vertical scrolling/overscroll
  useEffect(() => {
    if (typeof document === "undefined") return;

    const html = document.documentElement;
    const body = document.body;
    const origHtmlHeight = html.style.height;
    const origBodyHeight = body.style.height;
    const origBodyOverflow = body.style.overflowY;
    const origBodyOverscroll = body.style.overscrollBehavior;

    if (isMobile) {
      html.style.height = "100%";
      body.style.height = "100%";
      body.style.overflowY = "hidden";
      body.style.overscrollBehavior = "none";
    } else {
      body.style.overflowY = origBodyOverflow || "";
      body.style.overscrollBehavior = origBodyOverscroll || "";
    }

    const handleTouchMove = (e) => {
      if (isMobile && e.touches && e.touches.length > 1) {
        e.preventDefault();
      }
    };

    const handleGestureStart = (e) => {
      if (isMobile) e.preventDefault();
    };

    document.addEventListener("touchmove", handleTouchMove, { passive: false });
    document.addEventListener("gesturestart", handleGestureStart);

    return () => {
      document.removeEventListener("touchmove", handleTouchMove);
      document.removeEventListener("gesturestart", handleGestureStart);

      html.style.height = origHtmlHeight;
      body.style.height = origBodyHeight;
      body.style.overflowY = origBodyOverflow;
      body.style.overscrollBehavior = origBodyOverscroll;
    };
  }, [isMobile]);

  return (
    <div
      className="bg-black text-white"
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <Header />
      <Greeting />
      <CardStack isMobile={isMobile} />
    </div>
  );
}

function Header() {
  return (
    <div className="flex justify-between items-center px-4 py-3">
      <Lightbulb strokeWidth={1.5} color="#F9575C" className="w-9 h-9 justify-center"/>
      <div className="text-white text-xl font-extrabold">Lumos</div>
      <div className="w-10 h-10 rounded-full bg-white overflow-hidden">
        <img
          src="https://shapes.inc/api/public/avatar/minipekka-91nt"
          alt="Avatar"
          className="w-full h-full object-cover"
        />
      </div>
    </div>
  );
}

function Greeting() {
  return (
    <div className="px-6 sm:px-9 py-2 mt-6 overflow-y-hidden">
      <h1 className="text-[32px] sm:text-[36px] font-bold leading-5 mb-4" style={{ fontFamily: "Lato" }}>
        Hi, <span className="text-[#F9575C]">Tony Stark!</span>
      </h1>
      <div
        className="text-[30px] sm:text-[34px] leading-[34px]"
        style={{
          fontFamily: "Lato",
          textShadow: "0 4px 4px rgba(197, 197, 197, 0.35)",
        }}
      >
        <span className="font-light text-white">What do you want to do </span>
        <span className="font-bold text-[#57CEF9]">today?</span>
      </div>
    </div>
  );
}

function CardStack({ isMobile }) {
  const cards = [
    { id: 0, type: "ai" },
    { id: 1, type: "discover" },
    { id: 2, type: "photoroom" },
  ];
  const navigate = useNavigate();
  const { clickPosition, handleCardClick, setClickPosition } = useCardClickAnimation();
  const [clickedCardId, setClickedCardId] = useState(null);

  // touch/swipe state (only used on mobile)
  const [touchStartY, setTouchStartY] = useState(null);
  const [touchEndY, setTouchEndY] = useState(null);
  const [ismoveging, setIsmoveging] = useState(false);
  const [cooldown, setCooldown] = useState(false);

  const minSwipeDistance = 50; // how far user must swipe

  const onTouchMove = useCallback(
    (e) => {
      if (!isMobile) return;
      if (e.targetTouches && e.targetTouches[0]) {
        setTouchEndY(e.targetTouches[0].clientY);
      }
    },
    [isMobile]
  );

  const onTouchStart = useCallback(
    (e) => {
      if (!isMobile) return;
      setIsmoveging(true);
      setTouchEndY(null);
      if (e.targetTouches && e.targetTouches[0]) {
        setTouchStartY(e.targetTouches[0].clientY);
      }
    },
    [isMobile]
  );

  const onTouchEnd = useCallback(() => {
    if (!isMobile) {
      setIsmoveging(false);
      return;
    }

    if (!touchStartY || !touchEndY) {
      setTimeout(() => setIsmoveging(false), 150);
      return;
    }

    const distance = touchStartY - touchEndY;

    if (Math.abs(distance) >= minSwipeDistance && !cooldown) {
      setCooldown(true);
      if (distance > 0) goNext();
      else goPrev();
      setTimeout(() => setCooldown(false), 2000);
    }

    setTimeout(() => {
      setIsmoveging(false);
    }, 120);
  }, [isMobile, touchStartY, touchEndY, cooldown]);

  const [current, setCurrent] = useState(1);
  const n = cards.length;

  const goNext = useCallback(() => setCurrent((i) => (i + 1) % n), [n]);
  const goPrev = useCallback(() => setCurrent((i) => (i - 1 + n) % n), [n]);

  // auto-scroll only on mobile and only when not interacting
  useEffect(() => {
    if (!isMobile) return;
    if (ismoveging) return;

    const interval = setInterval(() => {
      goNext();
    }, 5000); // auto scroll speed for mobile

    return () => clearInterval(interval);
  }, [n, isMobile, ismoveging, goNext]);

  // container sizing: mobile uses vw / constrained width; desktop uses compact fixed width and center alignment
  const containerClass = isMobile
    ? "relative w-[85vw] max-w-[360px] h-[65vh] overflow-visible"
    : "relative w-[420px] h-[420px] flex items-center justify-center";

  return (
    <div className="relative flex-1 mt-8 flex px-4 items-center">
      <div className="flex-1 flex justify-end items-center relative">
        {/* arrow buttons (desktop + mobile) */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 flex flex-col items-center gap-4 z-50">
          <button
            onClick={goPrev}
            className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
            </svg>
          </button>
          <button
            onClick={goNext}
            className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {/* the stack */}
        <div
          className={containerClass}
          onTouchStart={isMobile ? onTouchStart : undefined}
          onTouchMove={isMobile ? onTouchMove : undefined}
          onTouchEnd={isMobile ? onTouchEnd : undefined}
          style={{
            touchAction: isMobile ? "pan-x" : "auto",
          }}
        >
          {cards.map((card, index) => {
            const rel = (index - current + n) % n;

            let translateY, rotate, scale, opacity, zIndex;

            // active
            if (rel === 0) {
              translateY = 0;
              rotate = 0;
              scale = 1;
              opacity = 1;
              zIndex = 40;
            }
            // next
            else if (rel === 1) {
              translateY = isMobile ? 160 : 130;
              rotate = -8;
              scale = isMobile ? 0.94 : 0.95;
              opacity = 0.95;
              zIndex = 30;
            }
            // previous
            else if (rel === n - 1) {
              translateY = isMobile ? -160 : -130;
              rotate = 8;
              scale = isMobile ? 0.94 : 0.95;
              opacity = 0.95;
              zIndex = 30;
            }
            // hidden ones
            else {
              translateY = isMobile ? 300 : 220;
              rotate = 0;
              scale = isMobile ? 0.8 : 0.88;
              opacity = 0;
              zIndex = 10;
            }

            return (
              <div
                key={card.id + "-" + index}
                style={{
                  position: "absolute",
                  top: "50%",
                  right: 0,
                  transform: `translateY(-50%) translate(${rel === 0 ? "-40px" : "0px"}, ${translateY}px) scale(${scale}) rotate(${rotate}deg)`,
                  opacity,
                  zIndex,
                  transition: ismoveging
                    ? "transform 0.25s ease-out, opacity 0.25s ease-out"
                    : "transform 0.8s cubic-bezier(0.9, -0.25, 0.1, 1.25), opacity 0.8s cubic-bezier(0.9, -0.25, 0.1, 1.25)",
                  pointerEvents: opacity === 0 ? "none" : "auto",
                  width: "100%",
                }}
              >
                <div
                  onClick={(e) => {
                    // On mobile we rely on ripple and navigate
                    handleCardClick(e);
                    setClickedCardId(card.id);
                    setTimeout(() => {
                      navigate(`/${card.type}`);
                      setClickPosition(null);
                      setClickedCardId(null);
                    }, 300);
                  }}
                  style={{ cursor: "pointer", position: "relative" }}
                >
                  {clickedCardId === card.id && (
                    <RippleEffect clickPosition={clickPosition} onAnimationEnd={() => setClickPosition(null)} />
                  )}
                  {renderCardComponent(card.type)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* -------------------------
   Card components (original heights preserved)
   ------------------------- */

function AICanvasCard({ isMobile }) {
  return (
    <div
      className="rounded-[13px] shadow-lg relative"
      style={{
        width: "100%",
        maxWidth: isMobile ? "332px" : "360px",
        height: isMobile ? "200px" : "220px", // original heights restored
        left: isMobile ? "70px" : "40px",
        background: isMobile ? "rgba(219, 39, 119, 0.95)" : "rgba(219,39,119,0.9)",
      }}
    >
      <div className="relative w-full h-full">
        <div className="absolute text-white font-light text-[20px] leading-[39px]" style={{ left: "16px", top: "16px", fontFamily: "Lato" }}>
          <img src={canvasicon} className="w-9 h-6 object-contain -mt-[5px]" alt="canvas" />
        </div>

        <div className="absolute text-white font-bold text-[25px] leading-6" style={{ left: "16px", top: "40px", textShadow: "0 4px 4px rgba(197, 197, 197, 0.35)", fontFamily: "Lato" }}>
          AI Canvas
        </div>

        <div className="absolute text-white font-light text-[20px] leading-[39px]" style={{ left: "16px", top: "56px", fontFamily: "Lato" }}>
          Shape your vision
        </div>

        <img
          src="https://api.builder.io/api/v1/image/assets/TEMP/a9515044985a444e694f021e50ab31495afbbad9?width=459"
          alt=""
          className="absolute rounded-lg"
          style={{
            width: "calc(100% - 20px)",
            height: "auto",
            transform: "rotate(-12deg)",
            left: "10px",
            top: "73px",
          }}
        />
      </div>
    </div>
  );
}

function DiscoverCard({ isMobile }) {
  return (
    <div
      className="rounded-[13px] shadow-lg relative"
      style={{
        width: isMobile ? "332px" : "360px",
        height: isMobile ? "200px" : "220px", // original heights restored
        background: "rgba(157, 87, 249, 1)",
        boxShadow: "0 4px 45px 3px rgba(181, 180, 180, 0.63)",
        left: isMobile ? "70px" : "40px",
      }}
    >
      <div className="relative w-full h-full">
        <div className="absolute text-white font-light text-[20px]" style={{ left: "16px", top: "16px", fontFamily: "Lato" }}>
          <img src={discovericon} className="w-9 h-6 object-contain -mt-[5px]" alt="discover" />
        </div>

        <div className="absolute text-white font-bold text-[25px] leading-6" style={{ left: "16px", top: "40px", textShadow: "0 4px 4px rgba(197,197,197,0.35)", fontFamily: "Lato" }}>
          Discover and Explore
        </div>

        <div className="absolute text-white font-light text-[20px] leading-[39px]" style={{ left: "16px", top: "56px", fontFamily: "Lato" }}>
          See what's trending
        </div>

        <div className="absolute flex gap-[10px]" style={{ left: "14px", top: "90px" }}>
          <img src="https://api.builder.io/api/v1/image/assets/TEMP/bff5d99682f1c0bc6f17392e24bf2fb20155cb4f?width=190" alt="" className="rounded-lg" style={{ width: "95px", height: "95px" }} />
          <img src="https://api.builder.io/api/v1/image/assets/TEMP/b67eea6fff7fbf7c0746315334564fd63a8e1481?width=190" alt="" className="rounded-lg" style={{ width: "95px", height: "95px" }} />
          <img src="https://api.builder.io/api/v1/image/assets/TEMP/1391b0f1b0032533a25185686d6c5ef2293effde?width=190" alt="" className="rounded-lg" style={{ width: "95px", height: "95px" }} />
        </div>
      </div>
    </div>
  );
}

function PhotoRoomCard({ isMobile }) {
  return (
    <div
      className="rounded-[13px] shadow-lg relative"
      style={{
        width: isMobile ? "332px" : "360px",
        height: isMobile ? "200px" : "220px", // original heights restored
        background: "rgba(87, 206, 249, 0.9)",
        left: isMobile ? "70px" : "40px",
      }}
    >
      <div className="relative w-full h-full">
        <div className="absolute text-white font-light text-[20px]" style={{ left: "16px", top: "16px", fontFamily: "Lato" }}>
          <img src={photoroomicon} className="w-9 h-6 object-contain -mt-[5px]" alt="photoroom" />
        </div>

        <div className="absolute text-white font-bold text-[25px] leading-6" style={{ left: "16px", top: "40px", textShadow: "0 4px 4px rgba(197,197,197,0.35)", fontFamily: "Lato" }}>
          Photo Room
        </div>

        <div className="absolute text-white font-light text-[20px] leading-[39px]" style={{ left: "16px", top: "56px", fontFamily: "Lato" }}>
          Snap or select
        </div>

        <img
          src="https://api.builder.io/api/v1/image/assets/TEMP/9f8a48c7915736e83f07376d86bf3e7b7e40148b?width=435"
          alt=""
          className="absolute rounded-lg"
          style={{
            width: "calc(100% - 20px)",
            height: "auto",
            transform: "rotate(12deg)",
            left: "10px",
            top: "60px",
          }}
        />
      </div>
    </div>
  );
}

// helper to render card components in CardStack (kept here to avoid undefined)
function renderCardComponent(type) {
  if (type === "ai") return <AICanvasCard isMobile={true} />;
  if (type === "discover") return <DiscoverCard isMobile={true} />;
  if (type === "photoroom") return <PhotoRoomCard isMobile={true} />;
  return null;
}

