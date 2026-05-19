import { useEffect, useState, useRef, RefObject } from 'react';

interface IntersectionObserverOptions extends IntersectionObserverInit {
  freezeOnceVisible?: boolean;
}

function useIntersectionObserver(
  elementRef: RefObject<Element>,
  {
    threshold = 0.1, // Trigger when 10% of the element is visible
    root = null, // Observe against the viewport
    rootMargin = '0%',
    freezeOnceVisible = true, // Animation runs once when element becomes visible
  }: IntersectionObserverOptions = {},
): boolean {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    // If we've already set it to intersecting and we want to freeze,
    // and an observer instance exists, disconnect it.
    if (observerRef.current && freezeOnceVisible && isIntersecting) {
      // No need to disconnect if it's already disconnected by unobserve
      // but good for explicit cleanup if the effect re-runs unexpectedly.
      return;
    }
    
    const node = elementRef?.current; 
    if (!node) return;

    // Ensure any old observer is disconnected before creating a new one
    if (observerRef.current) {
        observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsIntersecting(true);
          // If we want to freeze once visible, unobserve the target.
          if (freezeOnceVisible && observerRef.current) {
            observerRef.current.unobserve(entry.target);
            // Optionally disconnect fully if only observing one element ever with this instance
            // observerRef.current.disconnect(); 
          }
        } else {
          // Only set to false if we don't want to freeze
          if (!freezeOnceVisible) {
             setIsIntersecting(false);
          }
        }
      },
      { threshold, root, rootMargin }
    );

    observerRef.current.observe(node);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  // Add isIntersecting to dependencies only if freezeOnceVisible is false,
  // or manage disconnection more carefully.
  // For freezeOnceVisible=true, isIntersecting changing to true and then unobserving is the goal.
  }, [elementRef, threshold, root, rootMargin, freezeOnceVisible, isIntersecting]);

  return isIntersecting;
}

export default useIntersectionObserver;