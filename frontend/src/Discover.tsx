import "./discover.css";

import * as React from "react";
import * as ToastPrimitives from "@radix-ui/react-toast";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cva, type VariantProps } from "class-variance-authority";
import { X } from "lucide-react";
import { useTheme } from "next-themes";
import { Toaster as SonnerToaster } from "sonner";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router-dom";
import { motion, type Transition } from "framer-motion";
import { Bell, ChevronLeft, ArrowRight } from "lucide-react";
import { cn } from "./lib/utils.ts";

const queryClient = new QueryClient();

// ============ TOAST COMPONENTS ============
const ToastProvider = ToastPrimitives.Provider;

const ToastViewport = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Viewport>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Viewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Viewport
    ref={ref}
    className={cn(
      "fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]",
      className,
    )}
    {...props}
  />
));
ToastViewport.displayName = ToastPrimitives.Viewport.displayName;

const toastVariants = cva(
  "group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border p-6 pr-8 shadow-lg transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full",
  {
    variants: {
      variant: {
        default: "border bg-background text-foreground",
        destructive:
          "destructive group border-destructive bg-destructive text-destructive-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Root> &
    VariantProps<typeof toastVariants>
>(({ className, variant, ...props }, ref) => {
  return (
    <ToastPrimitives.Root
      ref={ref}
      className={cn(toastVariants({ variant }), className)}
      {...props}
    />
  );
});
Toast.displayName = ToastPrimitives.Root.displayName;

const ToastAction = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Action>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Action>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Action
    ref={ref}
    className={cn(
      "inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-sm font-medium ring-offset-background transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 group-[.destructive]:border-muted/40 group-[.destructive]:hover:border-destructive/30 group-[.destructive]:hover:bg-destructive group-[.destructive]:hover:text-destructive-foreground group-[.destructive]:focus:ring-destructive",
      className,
    )}
    {...props}
  />
));
ToastAction.displayName = ToastPrimitives.Action.displayName;

const ToastClose = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Close>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Close>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Close
    ref={ref}
    className={cn(
      "absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100 group-[.destructive]:text-red-300 group-[.destructive]:hover:text-red-50 group-[.destructive]:focus:ring-red-400 group-[.destructive]:focus:ring-offset-red-600",
      className,
    )}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitives.Close>
));
ToastClose.displayName = ToastPrimitives.Close.displayName;

const ToastTitle = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Title>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Title>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Title
    ref={ref}
    className={cn("text-sm font-semibold", className)}
    {...props}
  />
));
ToastTitle.displayName = ToastPrimitives.Title.displayName;

const ToastDescription = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Description>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Description>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Description
    ref={ref}
    className={cn("text-sm opacity-90", className)}
    {...props}
  />
));
ToastDescription.displayName = ToastPrimitives.Description.displayName;

type ToastProps = React.ComponentPropsWithoutRef<typeof Toast>;
type ToastActionElement = React.ReactElement<typeof ToastAction>;

// ============ TOASTER COMPONENT (Sonner) ============
type ToasterProps = React.ComponentProps<typeof SonnerToaster>;

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme();

  return (
    <SonnerToaster
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-muted-foreground",
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
        },
      }}
      {...props}
    />
  );
};

// ============ TOOLTIP COMPONENTS ============
const TooltipProvider = TooltipPrimitive.Provider;
const Tooltip = TooltipPrimitive.Root;
const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className,
    )}
    {...props}
  />
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

// ============ NOTFOUND COMPONENT ============
const NotFound = () => {
  const location = useLocation();

  React.useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname,
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">404</h1>
        <p className="text-xl text-gray-600 mb-4">Oops! Page not found</p>
        <a href="/" className="text-blue-500 hover:text-blue-700 underline">
          Return to Home
        </a>
      </div>
    </div>
  );
};

// ============ PAGE COMPONENTS ============
function Header() {
  const navigate = useNavigate();
  return (
    <motion.header 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex items-center justify-between px-4 sm:px-6 pt-8 pb-6"
    >
      <button
        onClick={() => navigate('/')}
        aria-label="Go back to home"
        className="flex items-center justify-center w-[30px] h-[30px] rounded-full border border-white/10 bg-[rgba(26,26,26,0.60)] shadow-[0_8px_32px_0_rgba(0,0,0,0.40)]"
      >
        <ChevronLeft className="w-4 h-4 text-[#666666]" />
      </button>

      <h1 className="flex-1 text-center text-xl font-semibold tracking-tight px-4">
        <span className="text-white">Good Evening, </span>
        <span className="text-[#F5576C]">Tony Stark!</span>
      </h1>

      <button className="flex items-center justify-center w-[30px] h-[30px] rounded-full border border-white/10 bg-[rgba(26,26,26,0.60)] shadow-[0_8px_32px_0_rgba(0,0,0,0.40)]">
        <Bell className="w-[15px] h-[15px] text-[#666666]" />
      </button>
    </motion.header>
  );
}

function TrendingSection() {
  const trendingItems = [
    {
      title: "CYBERPUNK CORE",
      image: "https://cdn.displate.com/artwork/270x380/2023-06-12/e33b59edee184d19f56aeba2f2c20186_817e95444bc40874b52bfcacfb6dc43d.jpg",
      bgColor: "rgba(173, 70, 255, 0.15)",
      borderColor: "rgba(173, 70, 255, 0.10)",
    },
    {
      title: "OCEAN COLLAGE",
      image: "https://i.pinimg.com/736x/94/1b/9c/941b9c6296e053d99225bf6b4442f0a0.jpg",
      bgColor: "rgba(107, 154, 225, 0.24)",
      borderColor: "rgba(43, 188, 255, 0.10)",
    },
    {
      title: "STAINED GLASS ART",
      image: "https://images.unsplash.com/photo-1632230997264-b2bfc65cb8b4?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8c3RhaW5lZCUyMGdsYXNzfGVufDB8fDB8fHww",
      bgColor: "rgba(255, 160, 43, 0.2)",
      borderColor: "rgba(225, 211, 107, 0.1)",
    },
    {
      title: "MORNING GLOW",
      image: "https://www.shutterbug.com/images/styles/960-wide/public/photo_post/%5Buid%5D/IMG_9710_filtereda.jpg",
      bgColor: "rgba(246, 51, 154, 0.15)",
      borderColor: "rgba(188, 51, 122, 0.10)",
    },
  ];

  return (
    <section className="mb-8">
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="flex items-center justify-between mb-4"
      >
        <h2 className="text-xl font-semibold tracking-tight">Trending Editing Styles</h2>
        <button className="text-[#666666]/65">
          <ArrowRight className="w-[18px] h-[18px]" />
        </button>
      </motion.div>

      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3 mb-12">
        {trendingItems.map((item, index) => (
          <motion.div
            key={item.title}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            // --- ANIMATION CHANGES START ---
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.95 }}
            transition={{
              // Spring gives it the bounce effect
              type: "spring" as const,
              stiffness: 400,
              damping: 17,
              // We keep the opacity delay for the entrance animation
              opacity: { delay: 0.3 + index * 0.1, duration: 0.4 }
            }}
            // --- ANIMATION CHANGES END ---
            className="relative h-[140px] sm:h-[140px] md:h-[160px] rounded-[10px] overflow-hidden cursor-pointer" // Added cursor-pointer
            style={{
              background: item.bgColor,
              border: `1px solid ${item.borderColor}`,
            }}
          >
            <img
              src={item.image}
              alt={item.title}
              className={`absolute ${
                index < 2 
                  ? "w-[125px] h-[180px] sm:w-[132px] sm:h-[200px] -rotate-90 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2" 
                  : "w-[180px] h-[130px] sm:w-[200px] sm:h-[132px] left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
              } rounded-[7px] object-cover shadow-sm`}
            />
            <div className="absolute bottom-2 right-2 text-right z-10">
              <p className="text-[12px] sm:text-s font-bold text-white/90 drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)] px-1 tracking-wide">
                {item.title}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
function NewSection() {
  const newItems = [
    {
      title: "AI Enhance",
      image: "https://api.builder.io/api/v1/image/assets/TEMP/41bea19f56d0af0f0665f03e7db78100eae3e795?width=204",
    },
    {
      title: "AI Stylization",
      image: "https://api.builder.io/api/v1/image/assets/TEMP/42712fe16f995bc383b29a72cd38816e818df6e1?width=204",
    },
    {
      title: "Potrait Studio",
      image: "https://images.stockcake.com/public/1/f/1/1f102fca-6a1a-42b0-8749-9ae6830716d3_large/intense-gaze-portrait-stockcake.jpg",
    },
  ];

  const galleryItems = [
    {
      title: "Polaroid Roll",
      subtitle: "~Brucey",
      image: "https://api.builder.io/api/v1/image/assets/TEMP/e6efa5244622730863c729b801edd6e0c55aed9b?width=204",
    },
    {
      title: "Rain Cinematic Shoot",
      subtitle: "~Wanda",
      image: "https://api.builder.io/api/v1/image/assets/TEMP/61fa026fac36d7c5d7f532b9d54a00cd93a931bd?width=204",
    },
    {
      title: "Saturday Journal Doodle",
      subtitle: "~Brucey",
      image: "https://images.stockcake.com/public/a/d/6/ad6a948a-b83e-4308-bfde-0b318aface70_large/rainy-window-view-stockcake.jpg",
    },
  ];

  // Animation config for the top row
  const tileAnimation = (index: number) => ({
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0 },
    whileHover: { scale: 1.02 },
    whileTap: { scale: 0.95 },
    transition: { 
      type: "spring" as const, 
      stiffness: 400, 
      damping: 17,
      opacity: { delay: 0.6 + index * 0.1, duration: 0.4 },
      x: { delay: 0.6 + index * 0.1, duration: 0.4 }
    }
  });

  // Animation config for the bottom row (delayed slightly longer to account for new header)
  const galleryAnimation = (index: number) => ({
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0 },
    whileHover: { scale: 1.02 },
    whileTap: { scale: 0.95 },
    transition: { 
      type: "spring" as const, 
      stiffness: 400, 
      damping: 17,
      opacity: { delay: 1.0 + index * 0.1, duration: 0.4 }, // Increased delay slightly
      x: { delay: 1.0 + index * 0.1, duration: 0.4 }
    }
  });

  return (
    <section className="pb-4">
      {/* --- First Header --- */}
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5, duration: 0.5 }}
        className="flex items-center justify-between mb-4"
      >
        <h2 className="text-xl font-semibold tracking-tight">
          <span className="text-white">See what's </span>
          <span className="text-[#FE5959]">NEW</span>
        </h2>
        <button className="text-[#666666]/65">
          <ArrowRight className="w-[18px] h-[18px]" />
        </button>
      </motion.div>

      {/* --- First Row (New Items) --- */}
      <div className="flex gap-3 overflow-x-auto scrollbar-hide mb-8 pb-2 md:grid md:grid-cols-3 md:overflow-visible">
        {newItems.map((item, index) => (
          <motion.div
            key={item.title}
            {...tileAnimation(index)}
            className="flex-shrink-0 w-[128px] h-[190px] sm:w-[140px] sm:h-[210px] md:w-auto rounded-[10px] border border-white/10 bg-[rgba(26,26,26,0.60)] shadow-[0_8px_32px_0_rgba(0,0,0,0.40)] overflow-hidden relative cursor-pointer"
          >
            <img
              src={item.image}
              alt={item.title}
              className="w-[110px] h-[155px] sm:w-[120px] sm:h-[180px] md:w-full md:h-auto md:aspect-[102/155] object-cover rounded-[7px] mx-auto mt-[9px]"
            />
            <p className="absolute bottom-0.5 left-0 right-0 text-center text-[11px] sm:text-xs font-bold text-white">
              {item.title}
            </p>
          </motion.div>
        ))}
      </div>

      {/* --- Second Header (New addition) --- */}
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.85, duration: 0.5 }} // Animates in after the first row
        className="flex items-center justify-between mb-4"
      >
        <h2 className="text-lg sm:text-xl font-semibold tracking-tight leading-tight pr-2">
          <span className="text-white">New Styles from the artists you follow</span>
          {/* Using a slightly smaller font for the long subtitle on mobile to prevent wrapping issues */}
        </h2>
        <button className="text-[#666666]/65 flex-shrink-0">
          <ArrowRight className="w-[18px] h-[18px]" />
        </button>
      </motion.div>

      {/* --- Second Row (Gallery Items) --- */}
      <div className="flex gap-3 overflow-x-auto scrollbar-hide pb-2 md:grid md:grid-cols-3 md:overflow-visible">
        {galleryItems.map((item, index) => (
          <motion.div
            key={item.title}
            {...galleryAnimation(index)}
            className="flex-shrink-0 w-[128px] h-[183px] sm:w-[140px] sm:h-[210px] md:w-auto rounded-[10px] border border-white/10 bg-[rgba(26,26,26,0.60)] shadow-[0_8px_32px_0_rgba(0,0,0,0.40)] overflow-hidden relative cursor-pointer"
          >
            <img
              src={item.image}
              alt={item.title}
              className="w-[110px] h-[155px] sm:w-[120px] sm:h-[180px] md:w-full md:h-auto md:aspect-[102/155] object-cover rounded-[7px] mx-auto mt-[9px]"
            />
            <div className="absolute bottom-1 left-0 right-0 text-center">
              <p className="text-[12px] sm:text-[8px] font-bold text-white">{item.title}</p>
              <p className="text-[8px] sm:text-[5px] font-light italic text-white">{item.subtitle}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
function BottomNavigation() {
  return (
    <motion.nav 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.5 }}
      className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-[rgba(18,18,18,0.95)] via-[rgba(18,18,18,0.80)] to-transparent pt-4 pb-4"
    >
      <div className="max-w-md mx-auto px-6 sm:max-w-lg md:max-w-xl lg:max-w-2xl">
        <div className="flex items-center justify-between h-[61px] px-4 sm:px-6 rounded-full bg-[rgba(26,26,26,0.80)] shadow-[0_8px_32px_0_rgba(0,0,0,0.40)]">
          <button className="flex items-center justify-center w-10 h-10 rounded-full hover:bg-white/5 transition-colors">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M13.75 19.25V11.9167C13.75 11.6736 13.6534 11.4404 13.4815 11.2685C13.3096 11.0966 13.0764 11 12.8333 11H9.16667C8.92355 11 8.69039 11.0966 8.51849 11.2685C8.34658 11.4404 8.25 11.6736 8.25 11.9167V19.25" stroke="#FE5959" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2.75 9.16666C2.74994 8.89997 2.80806 8.63648 2.92031 8.39457C3.03255 8.15265 3.19623 7.93814 3.39992 7.76599L9.81658 2.26599C10.1475 1.98632 10.5667 1.83289 11 1.83289C11.4333 1.83289 11.8525 1.98632 12.1834 2.26599L18.6001 7.76599C18.8038 7.93814 18.9674 8.15265 19.0797 8.39457C19.1919 8.63648 19.2501 8.89997 19.25 9.16666V17.4167C19.25 17.9029 19.0568 18.3692 18.713 18.713C18.3692 19.0568 17.9029 19.25 17.4167 19.25H4.58333C4.0971 19.25 3.63079 19.0568 3.28697 18.713C2.94315 18.3692 2.75 17.9029 2.75 17.4167V9.16666Z" stroke="#FE5959" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <button className="flex items-center justify-center w-10 h-10 rounded-full hover:bg-white/5 transition-colors">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19.25 19.25L15.2717 15.2717" stroke="#666666" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M10.0833 17.4167C14.1334 17.4167 17.4167 14.1334 17.4167 10.0833C17.4167 6.03325 14.1334 2.75 10.0833 2.75C6.03325 2.75 2.75 6.03325 2.75 10.0833C2.75 14.1334 6.03325 17.4167 10.0833 17.4167Z" stroke="#666666" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <button className="flex items-center justify-center w-12 h-12 rounded-full border border-white/20 bg-[rgba(254,89,89,0.90)] shadow-[0_4px_24px_0_rgba(254,89,89,0.40)] hover:bg-[rgba(254,89,89,1)] transition-colors">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4.58333 11H17.4167" stroke="white" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M11 4.58331V17.4166" stroke="white" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <button className="flex items-center justify-center w-10 h-10 rounded-full hover:bg-white/5 transition-colors">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M14.8867 7.11334L13.233 12.0734C13.143 12.3435 12.9914 12.5888 12.7901 12.7901C12.5888 12.9914 12.3435 13.143 12.0734 13.233L7.11334 14.8867L8.76701 9.92659C8.85701 9.65656 9.00865 9.41119 9.20992 9.20992C9.41119 9.00865 9.65656 8.85701 9.92659 8.76701L14.8867 7.11334Z" stroke="#666666" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M11 20.1666C16.0626 20.1666 20.1667 16.0626 20.1667 11C20.1667 5.93737 16.0626 1.83331 11 1.83331C5.93738 1.83331 1.83333 5.93737 1.83333 11C1.83333 16.0626 5.93738 20.1666 11 20.1666Z" stroke="#666666" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <button className="flex items-center justify-center w-10 h-10 rounded-full hover:bg-white/5 transition-colors">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M17.4167 19.25V17.4167C17.4167 16.4442 17.0304 15.5116 16.3427 14.8239C15.6551 14.1363 14.7225 13.75 13.75 13.75H8.25001C7.27755 13.75 6.34492 14.1363 5.65729 14.8239C4.96965 15.5116 4.58334 16.4442 4.58334 17.4167V19.25" stroke="#666666" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M11 10.0833C13.0251 10.0833 14.6667 8.44171 14.6667 6.41667C14.6667 4.39162 13.0251 2.75 11 2.75C8.97497 2.75 7.33334 4.39162 7.33334 6.41667C7.33334 8.44171 8.97497 10.0833 11 10.0833Z" stroke="#666666" strokeWidth="1.83333" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
    </motion.nav>
  );
}

function HomePage() {
  return (
    <div className="h-screen-safe w-full bg-[#121212] text-white overflow-y-auto overflow-x-hidden">
      <div className="max-w-md mx-auto relative pb-28 sm:max-w-lg md:max-w-xl lg:max-w-2xl min-h-full">
        <Header />
        
        <main className="px-4 sm:px-6 md:px-8">
          <TrendingSection />
          <NewSection />
        </main>

        <BottomNavigation />
      </div>
    </div>
  );
}

// ============ Discover COMPONENT ============
const Discover = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <ToastProvider>
        <ToastViewport />
        <Toaster />
      </ToastProvider>
      {/* Rely on the top-level Router (in index.js) — render the Discover content directly */}
      <HomePage />
    </TooltipProvider>
  </QueryClientProvider>
);

// ============ RENDER Discover ============
export default Discover;
