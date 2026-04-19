import { motion } from "framer-motion";

interface LevelOneLoadMoreButtonProps {
  label: string;
  glowClassName?: string;
  onClick: () => void;
}

export default function LevelOneLoadMoreButton({
  label,
  onClick,
}: LevelOneLoadMoreButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      className="rounded-full border border-white/30 bg-white/15 px-8 py-3 text-sm font-bold text-white shadow-lg backdrop-blur-md transition-colors hover:bg-white/25"
    >
      {label}
    </motion.button>
  );
}
