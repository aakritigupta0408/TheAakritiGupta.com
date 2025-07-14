import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function Aviator() {
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
            <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-blue-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◉
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              SOARING
              <br />
              <span className="gold-shimmer">AVIATOR</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              MASTERING THE SKIES THROUGH PRECISION, NAVIGATION, AND COURAGE
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
                FLIGHT
                <br />
                <span className="gold-shimmer">MASTERY</span>
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
                    <div className="w-12 h-12 bg-blue-500/20 rounded-sm flex items-center justify-center text-blue-400 text-xl">
                      ✈️
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        FLIGHT OPERATIONS
                      </h3>
                      <p className="text-white/60 text-sm">
                        Advanced pilot training & certification
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Comprehensive pilot training encompassing flight theory,
                    navigation systems, weather analysis, and emergency
                    procedures - developing systematic thinking and risk
                    management skills.
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
                    <div className="w-12 h-12 bg-blue-500/20 rounded-sm flex items-center justify-center text-blue-400 text-xl">
                      🧭
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        NAVIGATION SYSTEMS
                      </h3>
                      <p className="text-white/60 text-sm">
                        Instrument flying & GPS mastery
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Mastery of complex navigation systems, instrument flight
                    rules, and GPS technology - directly applicable to system
                    architecture and spatial reasoning in AI algorithms.
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
                    <div className="w-12 h-12 bg-blue-500/20 rounded-sm flex items-center justify-center text-blue-400 text-xl">
                      ⚡
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        DECISION MAKING
                      </h3>
                      <p className="text-white/60 text-sm">
                        Split-second critical choices
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Training in rapid decision-making under pressure, risk
                    assessment, and contingency planning - essential skills for
                    system reliability and incident response.
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
                <div className="text-8xl text-blue-400 mb-8">🛩️</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  FLIGHT ACHIEVEMENTS
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-blue-400 font-bold mb-2">
                      PPL
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      PRIVATE PILOT LICENSE
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-blue-400 font-bold mb-2">
                      100+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      FLIGHT HOURS
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-blue-400 font-bold mb-2">
                      5
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      AIRCRAFT TYPES
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-blue-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      SKY MASTERY
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
              SYSTEMS
              <br />
              <span className="gold-shimmer">THINKING</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              HOW AVIATION EXPERTISE ENHANCES TECHNICAL ARCHITECTURE
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
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🎛️
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                SYSTEMS INTEGRATION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Understanding complex aircraft systems translates to designing
                integrated ML pipelines where multiple components must work
                seamlessly together for optimal performance.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                ⚡
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                RAPID DECISION MAKING
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Aviation's emphasis on quick, accurate decisions under pressure
                directly enhances incident response, system debugging, and
                real-time optimization of production systems.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🧭
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                NAVIGATION & PLANNING
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Flight planning and navigation expertise enhances project
                roadmap development, technical architecture planning, and
                strategic system migrations.
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
            <div className="text-6xl text-blue-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              FLIGHT
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "Flying teaches you that preparation, systematic thinking, and
              staying calm under pressure aren't just helpful—they're essential.
              Every flight is a reminder that excellence comes from mastering
              fundamentals."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, AVIATOR & SYSTEMS ARCHITECT
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
                onClick={() => navigate("/talent/motorcyclist")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-purple-400/50 text-purple-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-purple-400 hover:bg-purple-400/10 transition-all duration-300"
              >
                MOTORCYCLIST
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/pianist")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-green-400/50 text-green-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-green-400 hover:bg-green-400/10 transition-all duration-300"
              >
                PIANIST
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      <ChatBot />
    </div>
  );
}
