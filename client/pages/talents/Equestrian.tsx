import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function Equestrian() {
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
            <div className="w-24 h-24 bg-gradient-to-r from-amber-500 to-amber-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◈
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              ELEGANT
              <br />
              <span className="gold-shimmer">EQUESTRIAN</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              MASTERING THE ART OF PARTNERSHIP, GRACE, AND EQUINE EXCELLENCE
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
                EQUESTRIAN
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
                    <div className="w-12 h-12 bg-amber-500/20 rounded-sm flex items-center justify-center text-amber-400 text-xl">
                      🐎
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        PARTNERSHIP & TRUST
                      </h3>
                      <p className="text-white/60 text-sm">
                        Human-equine communication
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Building deep partnerships with horses requires exceptional
                    communication skills, empathy, and trust-building - directly
                    applicable to team leadership and stakeholder management.
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
                    <div className="w-12 h-12 bg-amber-500/20 rounded-sm flex items-center justify-center text-amber-400 text-xl">
                      ⚖️
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        BALANCE & CONTROL
                      </h3>
                      <p className="text-white/60 text-sm">
                        Physical & mental equilibrium
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Maintaining perfect balance while controlling a powerful
                    animal develops exceptional multitasking abilities and
                    graceful handling of complex, dynamic situations.
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
                    <div className="w-12 h-12 bg-amber-500/20 rounded-sm flex items-center justify-center text-amber-400 text-xl">
                      👑
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        ELEGANT LEADERSHIP
                      </h3>
                      <p className="text-white/60 text-sm">
                        Grace under pressure
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Leading a 1,200-pound horse with subtle cues and quiet
                    confidence teaches elegant leadership - guiding powerful
                    teams with finesse rather than force.
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
                <div className="text-8xl text-amber-400 mb-8">🏇</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  RIDING ACHIEVEMENTS
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-amber-400 font-bold mb-2">
                      8+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      YEARS RIDING
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-amber-400 font-bold mb-2">
                      15+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      HORSES TRAINED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-amber-400 font-bold mb-2">
                      5
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      DISCIPLINES
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-amber-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      GRACE & ELEGANCE
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
              LEADERSHIP
              <br />
              <span className="gold-shimmer">TRANSLATION</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              HOW EQUESTRIAN EXCELLENCE ENHANCES TEAM LEADERSHIP
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
              <div className="w-16 h-16 bg-gradient-to-r from-amber-500 to-amber-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🤝
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                PARTNERSHIP BUILDING
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Building trust and communication with horses translates to
                exceptional team building, stakeholder engagement, and
                cross-functional collaboration in engineering teams.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-amber-500 to-amber-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                👑
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                ELEGANT LEADERSHIP
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Leading with subtle guidance rather than force creates more
                effective engineering teams and promotes innovation through
                collaborative problem-solving.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-amber-500 to-amber-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                ⚖️
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                DYNAMIC BALANCE
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Managing multiple variables while maintaining control translates
                to excellent project management and the ability to balance
                technical debt, feature development, and system stability.
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
            <div className="text-6xl text-amber-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              PARTNERSHIP
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "True leadership isn't about dominance—it's about partnership,
              trust, and bringing out the best in others. A great rider makes
              the horse want to perform, just like a great leader inspires their
              team."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, EQUESTRIAN & TECHNICAL LEADER
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
