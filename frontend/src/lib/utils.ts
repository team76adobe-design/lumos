/**
 * Utility function to combine classNames conditionally
 * Used for merging Tailwind CSS classes
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// export function cn(...classes) {
//   return classes
//     .flat()
//     .filter((cls) => typeof cls === "string" && cls.length > 0)
//     .join(" ");
// }
