import React from "react";
import { vi } from "vitest";

export async function createFramerMotionMock() {
  const stripMotionProps = (props: Record<string, unknown>) => {
    const {
      animate,
      exit,
      initial,
      layout,
      layoutId,
      transition,
      viewport,
      whileHover,
      whileInView,
      whileTap,
      ...rest
    } = props;

    void animate;
    void exit;
    void initial;
    void layout;
    void layoutId;
    void transition;
    void viewport;
    void whileHover;
    void whileInView;
    void whileTap;

    return rest;
  };

  const createMock =
    (tag: keyof React.JSX.IntrinsicElements) =>
    React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
      ({ children, ...props }, ref) =>
        React.createElement(tag, { ref, ...stripMotionProps(props) }, children),
    );

  const motion = new Proxy(
    {},
    {
      get: (_, tag: string) =>
        createMock((tag as keyof React.JSX.IntrinsicElements) || "div"),
    },
  );

  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    motion,
  };
}

export function installMatchMediaMock() {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}
