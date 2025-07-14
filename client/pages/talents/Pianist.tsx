import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function Pianist() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen tom-ford-gradient relative overflow-x-hidden">
      <Navigation />

      {/* Hero Section */}
      <section className="relative z-20 pt-32 pb-20">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-center mb-16"
          >
            <div className="w-24 h-24 bg-gradient-to-r from-green-500 to-green-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◑
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              MUSICAL
              <br />
              <span className="gold-shimmer">ARTISTRY</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              MASTERING HARMONY, RHYTHM, AND EXPRESSIVE PERFORMANCE
            </p>
          </motion.div>
        </div>
      </section>

      {/* Skills & Training Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid lg:grid-cols-2 gap-16 items-center"
          >
            <div>
              <h2 className="tom-ford-heading text-4xl text-white mb-8">
                MUSICAL
                <br />
                <span className="gold-shimmer">EXCELLENCE</span>
              </h2>
              <div className="space-y-8">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.1 }}
                  className="tom-ford-card p-6 rounded-sm"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-green-500/20 rounded-sm flex items-center justify-center text-green-400 text-xl">
                      🎹
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        TECHNICAL PRECISION
                      </h3>
                      <p className="text-white/60 text-sm">
                        Classical & contemporary mastery
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Years of rigorous piano training developing finger
                    dexterity, muscle memory, and the ability to coordinate
                    complex movements - directly enhancing coding speed and
                    accuracy.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.2 }}
                  className="tom-ford-card p-6 rounded-sm"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-green-500/20 rounded-sm flex items-center justify-center text-green-400 text-xl">
                      🎵
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        PATTERN RECOGNITION
                      </h3>
                      <p className="text-white/60 text-sm">
                        Harmonic structures & rhythm
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Understanding musical patterns, scales, and harmonic
                    progressions develops advanced pattern recognition skills
                    essential for algorithm design and data analysis.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 }}
                  className="tom-ford-card p-6 rounded-sm"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-green-500/20 rounded-sm flex items-center justify-center text-green-400 text-xl">
                      🎭
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        EXPRESSIVE ARTISTRY
                      </h3>
                      <p className="text-white/60 text-sm">
                        Emotional intelligence & creativity
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Translating emotion through musical expression develops
                    creative problem-solving and the ability to design elegant,
                    intuitive user experiences and interfaces.
                  </p>
                </motion.div>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="tom-ford-glass p-12 rounded-sm text-center">
                <div className="text-8xl text-green-400 mb-8">🎼</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  MUSICAL ACHIEVEMENTS
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-green-400 font-bold mb-2">
                      15+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      YEARS PLAYING
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-green-400 font-bold mb-2">
                      50+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      PIECES MASTERED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-green-400 font-bold mb-2">
                      8
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      MUSIC GENRES
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-green-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      ARTISTIC EXPRESSION
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Professional Connection Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="tom-ford-heading text-5xl text-white mb-8">
              CREATIVE
              <br />
              <span className="gold-shimmer">INTELLIGENCE</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              HOW MUSICAL TRAINING ENHANCES TECHNICAL CREATIVITY
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-green-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🎵
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                ALGORITHMIC THINKING
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Understanding musical structures and progressions enhances
                algorithm design, helping create more elegant and efficient
                machine learning models with natural flow.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-green-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🧠
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                COGNITIVE FLEXIBILITY
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Musical training enhances cognitive flexibility and
                multi-tasking abilities, enabling better handling of complex
                technical challenges and creative problem-solving approaches.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-green-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🎭
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                USER EXPERIENCE DESIGN
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Musical expression and emotional communication skills translate
                to designing more intuitive user interfaces and creating
                technology that truly resonates with users.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Philosophy Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="tom-ford-glass p-16 rounded-sm text-center"
          >
            <div className="text-6xl text-green-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              ARTISTIC
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "Music teaches you that perfection lies not in flawless execution,
              but in the harmony between technical precision and emotional
              expression. The same principle applies to elegant code—it should
              function beautifully and feel right."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, PIANIST & CREATIVE TECHNOLOGIST
            </div>
          </motion.div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="tom-ford-heading text-3xl text-white mb-8">
              EXPLORE MORE
              <br />
              <span className="gold-shimmer">TALENTS</span>
            </h2>
            <div className="flex justify-center gap-6 flex-wrap">
              <motion.button
                onClick={() => navigate("/talent/marksman")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-red-400/50 text-red-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-red-400 hover:bg-red-400/10 transition-all duration-300"
              >
                MARKSMAN
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/equestrian")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-amber-400/50 text-amber-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-amber-400 hover:bg-amber-400/10 transition-all duration-300"
              >
                EQUESTRIAN
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/aviator")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-blue-400/50 text-blue-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-blue-400 hover:bg-blue-400/10 transition-all duration-300"
              >
                AVIATOR
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/motorcyclist")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-purple-400/50 text-purple-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-purple-400 hover:bg-purple-400/10 transition-all duration-300"
              >
                MOTORCYCLIST
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      <ChatBot />
    </div>
  );
}
