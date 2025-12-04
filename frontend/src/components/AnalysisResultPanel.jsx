import React from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';

const AnalysisResultPanel = ({
  isVisible,
  isLoading,
  analysisResult,
  error,
  onClose
}) => {
  if (!isVisible) return null;

  return createPortal(
    <AnimatePresence>
      {isVisible && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 bg-black z-[9999]"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            initial={{ y: '100%', opacity: 0 }}
            animate={{ y: '0%', opacity: 1 }}
            exit={{ y: '100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300, duration: 0.5 }}
            className="fixed bottom-0 left-0 right-0 z-[10000] bg-[#2e2e2e] rounded-t-3xl shadow-2xl"
            style={{ maxHeight: '70vh' }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <h2 className="text-white text-lg font-semibold">Image Analysis</h2>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-white/10"
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(70vh - 60px)' }}>
              {/* Loading State */}
              {isLoading && (
                <div className="flex flex-col items-center justify-center py-12 space-y-4">
                  <Loader2 size={48} className="text-[#FE5959] animate-spin" />
                  <p className="text-white/70 text-sm">Analyzing image for defects...</p>
                </div>
              )}

              {/* Error State */}
              {error && !isLoading && (
                <div className="flex flex-col items-center justify-center py-12 space-y-4">
                  <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center">
                    <AlertTriangle size={32} className="text-red-500" />
                  </div>
                  <p className="text-red-400 text-sm text-center">{error}</p>
                </div>
              )}

              {/* Results */}
              {analysisResult && !isLoading && !error && (
                <div className="space-y-6">
                  {/* Detected Defects */}
                  {analysisResult.detected_defects && analysisResult.detected_defects.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-white/90 font-medium text-sm uppercase tracking-wider">
                        Detected Issues
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {analysisResult.detected_defects.map((defect, index) => (
                          <span
                            key={index}
                            className="px-3 py-1.5 bg-[#FE5959]/20 text-[#FE5959] rounded-full text-sm font-medium border border-[#FE5959]/30"
                          >
                            {defect}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* No Defects Found */}
                  {analysisResult.detected_defects && analysisResult.detected_defects.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-8 space-y-3">
                      <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
                        <CheckCircle size={32} className="text-green-500" />
                      </div>
                      <p className="text-green-400 text-sm">No defects detected!</p>
                    </div>
                  )}

                  {/* Detailed Analysis */}
                  {analysisResult.analysis && (
                    <div className="space-y-3">
                      <h3 className="text-white/90 font-medium text-sm uppercase tracking-wider">
                        Detailed Analysis
                      </h3>
                      <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                        <p className="text-white/80 text-sm leading-relaxed whitespace-pre-wrap">
                          {analysisResult.analysis}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
};

export default AnalysisResultPanel;
