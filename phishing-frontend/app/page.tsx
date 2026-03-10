"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import * as Toast from "@radix-ui/react-toast"
import * as Label from "@radix-ui/react-label"
import { ShieldCheckIcon, ShieldAlertIcon, ZapIcon, AlertTriangleIcon, LayersIcon, GlobeIcon } from "lucide-react"

// ── New API response type ─────────────────────────────
type Layer = {
  ran: boolean
  score: number | null
  prediction: "phishing" | "legitimate" | "skipped"
  threshold: number
  error?: string | null
}

type PredictionResponse = {
  url: string
  scheme: "https" | "http"
  verdict: "phishing" | "legitimate" | "suspicious" | "unknown"
  confidence: "very_high" | "high" | "medium" | "low" | "none"
  reason: string
  http_warning: boolean
  layer1: Layer
  layer2: Layer & { error?: string | null }
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/predict"

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut" as const },
  },
}

// ── Verdict config ────────────────────────────────────
const verdictConfig = {
  phishing: {
    label: "⚠️ Phishing Detected",
    badge: "High Risk",
    color: "from-red-600 to-rose-500",
    textColor: "text-red-500",
    borderColor: "border-red-500/30",
    bgColor: "bg-red-500/10",
    dot: "bg-red-500",
    icon: ShieldAlertIcon,
  },
  legitimate: {
    label: "✓ Legitimate Site",
    badge: "Safe",
    color: "from-emerald-600 to-teal-500",
    textColor: "text-emerald-500",
    borderColor: "border-emerald-500/30",
    bgColor: "bg-emerald-500/10",
    dot: "bg-emerald-500",
    icon: ShieldCheckIcon,
  },
  suspicious: {
    label: "⚡ Suspicious",
    badge: "Review",
    color: "from-amber-500 to-orange-500",
    textColor: "text-amber-500",
    borderColor: "border-amber-500/30",
    bgColor: "bg-amber-500/10",
    dot: "bg-amber-500",
    icon: AlertTriangleIcon,
  },
  unknown: {
    label: "? Unknown",
    badge: "Error",
    color: "from-slate-500 to-slate-400",
    textColor: "text-slate-400",
    borderColor: "border-slate-500/30",
    bgColor: "bg-slate-500/10",
    dot: "bg-slate-500",
    icon: AlertTriangleIcon,
  },
}

const confidenceLabel: Record<string, string> = {
  very_high: "Very High",
  high: "High",
  medium: "Medium",
  low: "Low",
  none: "None",
}

export default function HomePage() {
  const [url, setUrl] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<PredictionResponse | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [toastOpen, setToastOpen] = React.useState(false)
  const controllerRef = React.useRef<AbortController | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setResult(null)
    setError(null)

    const trimmed = url.trim()
    if (!trimmed) {
      setError("Please enter a URL to analyze.")
      setToastOpen(true)
      return
    }

    try {
      setLoading(true)
      controllerRef.current?.abort()
      controllerRef.current = new AbortController()

      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
        signal: controllerRef.current.signal,
      })

      const data = await res.json().catch(() => null)

      if (!res.ok) {
        const msg = (data && (data.detail || data.message)) || `Error: ${res.status}`
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg))
      }

      setResult(data as PredictionResponse)
    } catch (err: any) {
      if (err.name === "AbortError") return
      setError(err.message || "An unexpected error occurred.")
      setToastOpen(true)
    } finally {
      setLoading(false)
    }
  }

  const cfg = result ? verdictConfig[result.verdict] : null

  return (
    <Toast.Provider swipeDirection="right">
      <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        {/* Background glows */}
        <div className="pointer-events-none fixed inset-0">
          <div className="absolute -top-40 left-1/3 h-96 w-96 rounded-full bg-blue-600/10 blur-3xl" />
          <div className="absolute bottom-0 right-1/4 h-80 w-80 rounded-full bg-purple-600/10 blur-3xl" />
        </div>

        <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-12">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="w-full max-w-2xl space-y-6"
          >
            {/* Header */}
            <motion.div variants={itemVariants} className="space-y-2 text-center">
              <div className="flex items-center justify-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 p-2.5">
                  <ShieldCheckIcon className="h-6 w-6 text-white" />
                </div>
                <h1 className="text-4xl font-bold tracking-tight text-white md:text-5xl">
                  <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                    TLS Guard
                  </span>
                </h1>
              </div>
              <p className="text-sm text-slate-400">
                Two-layer AI phishing detection — TLS analysis + HTML content scanning
              </p>
            </motion.div>

            {/* Main Card */}
            <motion.div
              variants={itemVariants}
              className="rounded-2xl border border-slate-800/50 bg-slate-900/60 backdrop-blur-xl shadow-2xl"
            >
              <div className="p-8 space-y-6">

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label.Root className="block text-sm font-semibold text-slate-300">
                      Enter URL to Analyze
                    </Label.Root>
                    <div className="relative">
                      <input
                        type="text"
                        placeholder="https://example.com/login"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        disabled={loading}
                        className="w-full rounded-xl border-2 border-slate-700 bg-slate-800/50 px-4 py-3.5 text-base text-slate-100 placeholder-slate-500 outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50"
                      />
                      <GlobeIcon className="pointer-events-none absolute right-4 top-3.5 h-5 w-5 text-slate-500" />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-3.5 font-semibold text-white shadow-lg transition-all hover:from-blue-700 hover:to-purple-700 hover:shadow-blue-600/30 disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    <div className="flex items-center justify-center gap-3">
                      {loading ? (
                        <>
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                            className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white"
                          />
                          <span>Scanning both layers...</span>
                        </>
                      ) : (
                        <>
                          <ZapIcon className="h-5 w-5" />
                          <span>Scan URL</span>
                        </>
                      )}
                    </div>
                  </button>
                </form>

                {/* Result */}
                <AnimatePresence>
                  {result && !loading && cfg && (
                    <motion.div
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 16 }}
                      transition={{ duration: 0.4 }}
                      className="space-y-5 border-t border-slate-800 pt-6"
                    >

                      {/* Verdict header */}
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <p className="text-xs font-bold uppercase tracking-widest text-slate-500">
                            Final Verdict
                          </p>
                          <div className="flex items-center gap-3">
                            <div className={`h-3 w-3 rounded-full ${cfg.dot}`} />
                            <p className={`text-2xl font-bold ${cfg.textColor}`}>
                              {cfg.label}
                            </p>
                          </div>
                          <p className="text-sm text-slate-400 mt-1">{result.reason}</p>
                        </div>
                        <div className="flex flex-col items-end gap-2 shrink-0">
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 200, damping: 15 }}
                            className={`rounded-full bg-gradient-to-br ${cfg.color} px-4 py-1.5 text-sm font-bold text-white shadow-lg`}
                          >
                            {cfg.badge}
                          </motion.div>
                          <span className="text-xs text-slate-500">
                            Confidence: <span className="text-slate-300 font-medium">{confidenceLabel[result.confidence]}</span>
                          </span>
                        </div>
                      </div>

                      {/* HTTP Warning */}
                      {result.http_warning && (
                        <div className="flex items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3">
                          <AlertTriangleIcon className="h-4 w-4 text-amber-400 shrink-0" />
                          <p className="text-sm text-amber-300">
                            This site uses HTTP — your connection is not encrypted
                          </p>
                        </div>
                      )}

                      {/* Layer breakdown */}
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <LayersIcon className="h-4 w-4 text-slate-500" />
                          <p className="text-xs font-bold uppercase tracking-widest text-slate-500">
                            Layer Breakdown
                          </p>
                        </div>

                        {/* Layer 1 */}
                        <div className={`rounded-xl border ${result.layer1.ran ? (result.layer1.prediction === "phishing" ? "border-red-500/30 bg-red-500/5" : "border-emerald-500/30 bg-emerald-500/5") : "border-slate-700/50 bg-slate-800/30"} p-4`}>
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-sm font-semibold text-slate-300">
                                Layer 1 — TLS / URL Features
                              </p>
                              <p className="text-xs text-slate-500 mt-0.5">
                                {result.scheme === "https" ? `Threshold: ${result.layer1.threshold}` : "Skipped — HTTP site (no TLS)"}
                              </p>
                            </div>
                            <div className="text-right">
                              {result.layer1.ran ? (
                                <>
                                  <p className={`text-sm font-bold ${result.layer1.prediction === "phishing" ? "text-red-400" : "text-emerald-400"}`}>
                                    {result.layer1.prediction === "phishing" ? "Phishing" : "Legitimate"}
                                  </p>
                                  <p className="text-xs text-slate-500">
                                    Score: {result.layer1.score?.toFixed(3)}
                                  </p>
                                </>
                              ) : (
                                <p className="text-sm text-slate-500 italic">Skipped</p>
                              )}
                            </div>
                          </div>
                          {/* Score bar */}
                          {result.layer1.ran && result.layer1.score !== null && (
                            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-700">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${result.layer1.score * 100}%` }}
                                transition={{ duration: 0.8, ease: "easeOut" }}
                                className={`h-full bg-gradient-to-r ${result.layer1.prediction === "phishing" ? "from-red-600 to-rose-500" : "from-emerald-600 to-teal-500"}`}
                              />
                            </div>
                          )}
                        </div>

                        {/* Layer 2 */}
                        <div className={`rounded-xl border ${result.layer2.ran ? (result.layer2.prediction === "phishing" ? "border-red-500/30 bg-red-500/5" : "border-emerald-500/30 bg-emerald-500/5") : "border-slate-700/50 bg-slate-800/30"} p-4`}>
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-sm font-semibold text-slate-300">
                                Layer 2 — HTML + URL Heuristics
                              </p>
                              <p className="text-xs text-slate-500 mt-0.5">
                                Threshold: {result.layer2.threshold}
                              </p>
                            </div>
                            <div className="text-right">
                              {result.layer2.ran ? (
                                <>
                                  <p className={`text-sm font-bold ${result.layer2.prediction === "phishing" ? "text-red-400" : "text-emerald-400"}`}>
                                    {result.layer2.prediction === "phishing" ? "Phishing" : "Legitimate"}
                                  </p>
                                  <p className="text-xs text-slate-500">
                                    Score: {result.layer2.score?.toFixed(3)}
                                  </p>
                                </>
                              ) : (
                                <p className="text-sm text-red-400 text-sm">{result.layer2.error || "Failed"}</p>
                              )}
                            </div>
                          </div>
                          {/* Score bar */}
                          {result.layer2.ran && result.layer2.score !== null && (
                            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-700">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${result.layer2.score * 100}%` }}
                                transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
                                className={`h-full bg-gradient-to-r ${result.layer2.prediction === "phishing" ? "from-red-600 to-rose-500" : "from-emerald-600 to-teal-500"}`}
                              />
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Analyzed URL */}
                      <div className="rounded-lg bg-slate-800/50 px-4 py-3">
                        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
                          Analyzed URL
                        </p>
                        <p className="break-all font-mono text-sm text-slate-300">{result.url}</p>
                      </div>

                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>

            {/* Stats */}
            <motion.div variants={itemVariants} className="grid grid-cols-3 gap-4">
              {[
                { label: "Layer 1", value: "TLS/URL", sub: "threshold 0.25" },
                { label: "Layer 2", value: "HTML+URL", sub: "threshold 0.20" },
                { label: "Combined", value: "ROC-AUC", sub: "0.9705" },
              ].map((stat, idx) => (
                <div
                  key={idx}
                  className="rounded-xl border border-slate-800/50 bg-slate-900/40 backdrop-blur px-4 py-4 text-center"
                >
                  <p className="text-xs text-slate-500 mb-1">{stat.label}</p>
                  <p className="font-bold text-white text-base">{stat.value}</p>
                  <p className="text-xs text-slate-600 mt-0.5">{stat.sub}</p>
                </div>
              ))}
            </motion.div>

          </motion.div>
        </main>

        {/* Error Toast */}
        <Toast.Root
          open={toastOpen && !!error}
          onOpenChange={setToastOpen}
          duration={4000}
          className="fixed bottom-6 right-6 w-96 rounded-xl border-2 border-red-900/50 bg-slate-900 p-4 shadow-2xl"
        >
          <div className="flex items-start gap-3">
            <div className="h-10 w-10 rounded-lg bg-red-900/20 flex items-center justify-center shrink-0">
              <ShieldAlertIcon className="h-5 w-5 text-red-400" />
            </div>
            <div className="flex-1">
              <Toast.Title className="font-bold text-white">Error</Toast.Title>
              <Toast.Description className="mt-1 text-sm text-slate-400">
                {error}
              </Toast.Description>
            </div>
          </div>
          <Toast.Close className="absolute right-4 top-4 text-slate-500 hover:text-slate-300" />
        </Toast.Root>
        <Toast.Viewport className="fixed bottom-0 right-0 flex flex-col gap-2 p-4" />

      </div>
    </Toast.Provider>
  )
}