import React from "react";
import { motion } from "framer-motion";

/**
 * PageTransition wrapper component that adds smooth, snappy animations
 * to page changes triggered by React Router
 */
export function PageTransition({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: -20 }}
      transition={{
        duration: 0.5,
        ease: [0.16, 1, 0.3, 1], // custom snappy cubic-bezier
      }}
      style={{ width: "100%", height: "100%" }}
    >
      {children}
    </motion.div>
  );
}

/**
 * Card click animation trigger - creates a ripple/burst effect
 * when card is clicked before navigation
 */
export function useCardClickAnimation() {
  const [clickPosition, setClickPosition] = React.useState(null);

  const handleCardClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setClickPosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return { clickPosition, handleCardClick, setClickPosition };
}

/**
 * Ripple effect component that expands from click point
 */
export function RippleEffect({ clickPosition, onAnimationEnd }) {
  if (!clickPosition) return null;

  return (
    <motion.div
      initial={{ scale: 0, opacity: 0.8 }}
      animate={{ scale: 4, opacity: 0 }}
      onAnimationComplete={onAnimationEnd}
      transition={{
        duration: 0.6,
        ease: [0.16, 1, 0.3, 1],
      }}
      style={{
        position: "absolute",
        left: clickPosition.x,
        top: clickPosition.y,
        width: 40,
        height: 40,
        marginLeft: -20,
        marginTop: -20,
        borderRadius: "50%",
        background:
          "radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 100%)",
        pointerEvents: "none",
        zIndex: 1000,
      }}
    />
  );
}

/**
 * Page container with staggered children animations
 */
export function AnimatedPageContainer({ children, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.4,
        delay: delay + 0.1,
        ease: [0.16, 1, 0.3, 1],
      }}
      style={{ width: "100%", height: "100%" }}
    >
      {children}
    </motion.div>
  );
}
