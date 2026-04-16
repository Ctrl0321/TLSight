"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import * as Toast from "@radix-ui/react-toast"
import * as Label from "@radix-ui/react-label"
import * as Tabs from "@radix-ui/react-tabs"
import {
  ShieldCheckIcon, ShieldAlertIcon, ZapIcon, AlertTriangleIcon,
  LayersIcon, GlobeIcon, UploadIcon, LinkIcon, XCircleIcon,
  CheckCircle2Icon, ClockIcon, ChevronDownIcon, ChevronUpIcon,
} from "lucide-react"


type Layer = {
  ran: boolean
  skipped?: boolean
  skip_reason?: string | null
  score: number | null
  prediction: "phishing" | "legitimate" | "skipped"
  threshold: number
  error?: string | null
}

type PredictionResponse = {
  input_url: string
  url: string
  scheme: "https" | "http" | null
  verdict: "phishing" | "legitimate" | "suspicious" | "unknown" | "invalid"
  confidence: "very_high" | "high" | "medium" | "low" | "none"
  reason: string
  http_warning: boolean
  layer1: Layer
  layer2: Layer & { error?: string | null; final_url?: string; model_url?: string }
  error?: string | null
  _index?: number
  type?: string
}

type BulkMeta = { type: "meta"; total: number }
type BulkDone = { type: "done"; total: number }
type StreamLine = PredictionResponse | BulkMeta | BulkDone


const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } },
}
const itemVariants = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" as const } },
}


const verdictConfig = {
  phishing: {
    label: "Phishing Detected",
    badge: "High Risk",
    emoji: "⚠️",
    color: "from-red-600 to-rose-500",
    textColor: "text-red-400",
    borderColor: "border-red-500/30",
    bgColor: "bg-red-500/10",
    dot: "bg-red-400",
    icon: ShieldAlertIcon,
  },
  legitimate: {
    label: "Legitimate Site",
    badge: "Safe",
    emoji: "✓",
    color: "from-emerald-600 to-teal-500",
    textColor: "text-emerald-400",
    borderColor: "border-emerald-500/30",
    bgColor: "bg-emerald-500/10",
    dot: "bg-emerald-400",
    icon: ShieldCheckIcon,
  },
  suspicious: {
    label: "Suspicious",
    badge: "Review",
    emoji: "⚡",
    color: "from-amber-500 to-orange-500",
    textColor: "text-amber-400",
    borderColor: "border-amber-500/30",
    bgColor: "bg-amber-500/10",
    dot: "bg-amber-400",
    icon: AlertTriangleIcon,
  },
  unknown: {
    label: "Unknown",
    badge: "Error",
    emoji: "?",
    color: "from-slate-500 to-slate-400",
    textColor: "text-slate-400",
    borderColor: "border-slate-600/30",
    bgColor: "bg-slate-800/40",
    dot: "bg-slate-400",
    icon: AlertTriangleIcon,
  },
  invalid: {
    label: "Invalid URL",
    badge: "Invalid",
    emoji: "✗",
    color: "from-slate-600 to-slate-500",
    textColor: "text-slate-400",
    borderColor: "border-slate-600/30",
    bgColor: "bg-slate-800/40",
    dot: "bg-slate-500",
    icon: XCircleIcon,
  },
}

const confidenceLabel: Record<string, string> = {
  very_high: "Very High",
  high: "High",
  medium: "Medium",
  low: "Low",
  none: "None",
}


function ScoreBar({ score, verdict, delay = 0 }: {
  score: number; verdict: string; delay?: number
}) {
  return (
    <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-700/60">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${score * 100}%` }}
        transition={{ duration: 0.9, ease: "easeOut", delay }}
        className={`h-full bg-gradient-to-r ${verdict === "phishing" ? "from-red-600 to-rose-500" : "from-emerald-600 to-teal-500"}`}
      />
    </div>
  )
}

function LayerCard({ title, subtitle, layer }: {
  title: string; subtitle: string; layer: Layer
}) {
  const isPhishing = layer.prediction === "phishing"
  const skipped    = layer.skipped || !layer.ran

  let borderCls = "border-slate-700/50 bg-slate-800/30"
  if (!skipped) {
    borderCls = isPhishing
      ? "border-red-500/25 bg-red-500/5"
      : "border-emerald-500/25 bg-emerald-500/5"
  }

  return (
    <div className={`rounded-xl border ${borderCls} p-4 transition-colors`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-200">{title}</p>
          <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{subtitle}</p>
        </div>
        <div className="text-right shrink-0">
          {skipped ? (
            <p className="text-xs text-slate-500 italic">Skipped</p>
          ) : layer.error ? (
            <p className="text-xs text-red-400">{layer.error}</p>
          ) : (
            <>
              <p className={`text-sm font-bold ${isPhishing ? "text-red-400" : "text-emerald-400"}`}>
                {isPhishing ? "Phishing" : "Legitimate"}
              </p>
              <p className="text-xs text-slate-500">Score: {layer.score?.toFixed(3)}</p>
            </>
          )}
        </div>
      </div>

      {/* Skip reason banner */}
      {skipped && layer.skip_reason && (
        <div className="mt-3 flex items-start gap-2 rounded-lg bg-slate-700/30 px-3 py-2">
          <AlertTriangleIcon className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />
          <p className="text-xs text-slate-400 leading-relaxed">{layer.skip_reason}</p>
        </div>
      )}

      {/* Score bar */}
      {!skipped && layer.score !== null && (
        <ScoreBar score={layer.score} verdict={layer.prediction} delay={0.15} />
      )}
    </div>
  )
}

function ResultCard({ result, compact = false }: {
  result: PredictionResponse; compact?: boolean
}) {
  const cfg = verdictConfig[result.verdict] ?? verdictConfig.unknown
  const Icon = cfg.icon
  const [expanded, setExpanded] = React.useState(false)

  return (
    <div className={`rounded-xl border ${cfg.borderColor} ${cfg.bgColor} overflow-hidden`}>
      {/* Header row */}
      <div className="flex items-center gap-3 px-4 py-3">
        <div className={`h-2.5 w-2.5 rounded-full shrink-0 ${cfg.dot}`} />
        <div className="min-w-0 flex-1">
          <p className="text-xs font-mono text-slate-500 truncate">{result.input_url}</p>
          <div className="flex items-center gap-2 mt-0.5">
            <span className={`text-sm font-bold ${cfg.textColor}`}>
              {cfg.emoji} {cfg.label}
            </span>
            {result.scheme && (
              <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                result.scheme === "https"
                  ? "bg-emerald-500/15 text-emerald-400"
                  : "bg-amber-500/15 text-amber-400"
              }`}>
                {result.scheme.toUpperCase()}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className={`text-xs px-2 py-1 rounded-full bg-gradient-to-r ${cfg.color} text-white font-semibold`}>
            {cfg.badge}
          </span>
          {!compact && (
            <button
              onClick={() => setExpanded(v => !v)}
              className="text-slate-500 hover:text-slate-300 transition-colors"
              aria-label="Toggle details"
            >
              {expanded
                ? <ChevronUpIcon className="h-4 w-4" />
                : <ChevronDownIcon className="h-4 w-4" />}
            </button>
          )}
        </div>
      </div>

      {/* Expandable detail */}
      <AnimatePresence initial={false}>
        {(expanded || compact) && !result.error && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1 space-y-3 border-t border-slate-700/40">
              <p className="text-xs text-slate-400 leading-relaxed">{result.reason}</p>

              {result.http_warning && (
                <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2">
                  <AlertTriangleIcon className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                  <p className="text-xs text-amber-300">
                    HTTP only — connection is unencrypted, TLS analysis skipped
                  </p>
                </div>
              )}

              <div className="grid grid-cols-1 gap-2">
                <LayerCard
                  title="Layer 1 — TLS / URL Features"
                  subtitle={
                    result.layer1.skipped
                      ? "Not run"
                      : `Threshold: ${result.layer1.threshold}`
                  }
                  layer={result.layer1}
                />
                <LayerCard
                  title="Layer 2 — HTML + URL Heuristics"
                  subtitle={`Threshold: ${result.layer2.threshold}`}
                  layer={result.layer2 as Layer}
                />
              </div>

              <div className="rounded-lg bg-slate-800/40 px-3 py-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
                  Confidence
                </p>
                <p className="text-sm text-slate-300 font-medium">
                  {confidenceLabel[result.confidence] ?? result.confidence}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      {result.error && (
        <div className="px-4 pb-3">
          <p className="text-xs text-red-400">{result.error}</p>
        </div>
      )}
    </div>
  )
}


function SinglePanel() {
  const [url, setUrl]       = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<PredictionResponse | null>(null)
  const [error, setError]   = React.useState<string | null>(null)
  const [toastOpen, setToastOpen] = React.useState(false)
  const ctrlRef = React.useRef<AbortController | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setResult(null); setError(null)
    const trimmed = url.trim()
    if (!trimmed) { setError("Please enter a URL."); setToastOpen(true); return }

    try {
      setLoading(true)
      ctrlRef.current?.abort()
      ctrlRef.current = new AbortController()
      const res  = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
        signal: ctrlRef.current.signal,
      })
      const data = await res.json().catch(() => null)
      if (!res.ok) {
        const msg = (data && (data.detail || data.message)) || `HTTP ${res.status}`
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg))
      }
      setResult(data as PredictionResponse)
    } catch (err: any) {
      if (err.name === "AbortError") return
      setError(err.message || "Unexpected error.")
      setToastOpen(true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label.Root className="block text-sm font-semibold text-slate-300">
            URL to Analyze
          </Label.Root>
          <div className="relative">
            <GlobeIcon className="pointer-events-none absolute left-4 top-3.5 h-5 w-5 text-slate-500" />
            <input
              type="text"
              placeholder="example.com  or  https://example.com/login"
              value={url}
              onChange={e => setUrl(e.target.value)}
              disabled={loading}
              className="w-full rounded-xl border-2 border-slate-700 bg-slate-800/50 pl-11 pr-4 py-3.5 text-base text-slate-100 placeholder-slate-500 outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50"
            />
          </div>
          <p className="text-xs text-slate-600">
            You can omit <code className="text-slate-500">https://</code> — the backend will probe HTTPS first,
            then HTTP. Invalid schemes (e.g. <code className="text-slate-500">ftp://</code>) are rejected.
          </p>
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
                  transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
                  className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white"
                />
                <span>Scanning…</span>
              </>
            ) : (
              <><ZapIcon className="h-5 w-5" /><span>Scan URL</span></>
            )}
          </div>
        </button>
      </form>

      <AnimatePresence>
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 14 }}
            transition={{ duration: 0.35 }}
          >
            <ResultCard result={result} compact />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toast */}
      <Toast.Root
        open={toastOpen && !!error}
        onOpenChange={setToastOpen}
        duration={4000}
        className="fixed bottom-6 right-6 w-96 rounded-xl border-2 border-red-900/50 bg-slate-900 p-4 shadow-2xl z-50"
      >
        <div className="flex items-start gap-3">
          <div className="h-10 w-10 rounded-lg bg-red-900/20 flex items-center justify-center shrink-0">
            <ShieldAlertIcon className="h-5 w-5 text-red-400" />
          </div>
          <div className="flex-1">
            <Toast.Title className="font-bold text-white">Error</Toast.Title>
            <Toast.Description className="mt-1 text-sm text-slate-400">{error}</Toast.Description>
          </div>
        </div>
        <Toast.Close className="absolute right-4 top-4 text-slate-500 hover:text-slate-300" />
      </Toast.Root>
      <Toast.Viewport className="fixed bottom-0 right-0 flex flex-col gap-2 p-4 z-50" />
    </div>
  )
}

function BulkPanel() {
  const [text, setText]     = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [results, setResults] = React.useState<PredictionResponse[]>([])
  const [total, setTotal]   = React.useState(0)
  const [done, setDone]     = React.useState(false)
  const [error, setError]   = React.useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const urls = text.split("\n").map(s => s.trim()).filter(Boolean)
    if (!urls.length) { setError("Enter at least one URL."); return }
    if (urls.length > 500) { setError("Maximum 500 URLs."); return }

    setResults([]); setError(null); setDone(false); setTotal(urls.length)
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/predict/bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls }),
      })
      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        throw new Error(d.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setResults(data.results)
      setDone(true)
    } catch (err: any) {
      setError(err.message || "Unexpected error.")
    } finally {
      setLoading(false)
    }
  }

  const phishCount = results.filter(r => r.verdict === "phishing").length
  const safeCount  = results.filter(r => r.verdict === "legitimate").length

  return (
    <div className="space-y-5">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label.Root className="block text-sm font-semibold text-slate-300">
            URLs — one per line (max 500)
          </Label.Root>
          <textarea
            rows={6}
            placeholder={"example.com\nhttps://suspicious-login.net\nanother-site.org"}
            value={text}
            onChange={e => setText(e.target.value)}
            disabled={loading}
            className="w-full rounded-xl border-2 border-slate-700 bg-slate-800/50 px-4 py-3 text-sm text-slate-100 placeholder-slate-600 outline-none font-mono resize-y transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50"
          />
          <p className="text-xs text-slate-600">
            Schemes are auto-detected. Results appear as each URL is processed.
          </p>
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-3.5 font-semibold text-white shadow-lg transition-all hover:from-blue-700 hover:to-purple-700 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <div className="flex items-center justify-center gap-3">
            {loading ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
                  className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white"
                />
                <span>Scanning {results.length} / {total}…</span>
              </>
            ) : (
              <><LinkIcon className="h-5 w-5" /><span>Scan All URLs</span></>
            )}
          </div>
        </button>
      </form>

      {/* Progress + summary */}
      {(loading || done) && total > 0 && (
        <div className="space-y-2">
          {/* Progress bar */}
          <div className="h-1.5 w-full rounded-full bg-slate-700 overflow-hidden">
            <motion.div
              animate={{ width: `${(results.length / total) * 100}%` }}
              transition={{ duration: 0.3 }}
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            />
          </div>
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>{results.length} / {total} completed</span>
            {done && (
              <span className="flex items-center gap-1 text-emerald-400">
                <CheckCircle2Icon className="h-3.5 w-3.5" /> All done
              </span>
            )}
          </div>

          {done && results.length > 0 && (
            <div className="grid grid-cols-3 gap-3 pt-1">
              {[
                { label: "Phishing",   count: phishCount,                         color: "text-red-400"     },
                { label: "Legitimate", count: safeCount,                          color: "text-emerald-400" },
                { label: "Other",      count: results.length - phishCount - safeCount, color: "text-slate-400"   },
              ].map(s => (
                <div key={s.label} className="rounded-lg bg-slate-800/50 px-3 py-2 text-center">
                  <p className={`text-lg font-bold ${s.color}`}>{s.count}</p>
                  <p className="text-xs text-slate-500">{s.label}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Result list — shown progressively */}
      {results.length > 0 && (
        <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
          <AnimatePresence initial={false}>
            {results.map((r, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
              >
                <ResultCard result={r} />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
function CSVPanel() {
  const [file, setFile]     = React.useState<File | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [results, setResults] = React.useState<PredictionResponse[]>([])
  const [total, setTotal]   = React.useState(0)
  const [done, setDone]     = React.useState(false)
  const [error, setError]   = React.useState<string | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  const handleFile = (f: File | null) => {
    if (!f) return
    if (!f.name.endsWith(".csv")) { setError("Please upload a .csv file."); return }
    setFile(f); setError(null); setResults([]); setDone(false); setTotal(0)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    handleFile(e.dataTransfer.files[0] ?? null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) { setError("Please select a CSV file."); return }

    setResults([]); setDone(false); setTotal(0); setError(null)
    setLoading(true)

    const fd = new FormData()
    fd.append("file", file)

    try {
      const res = await fetch(`${API_BASE}/predict/csv`, { method: "POST", body: fd })

      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        throw new Error(d.detail || `HTTP ${res.status}`)
      }

      // ── Stream NDJSON ────────────────────────────────
      const reader  = res.body!.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ""

      while (true) {
        const { done: streamDone, value } = await reader.read()
        if (streamDone) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""   // keep incomplete last chunk

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue
          try {
            const obj: StreamLine = JSON.parse(trimmed)
            if (obj.type === "meta") {
              setTotal((obj as BulkMeta).total)
            } else if (obj.type === "done") {
              setDone(true)
            } else if (obj.type === "result") {
              setResults(prev => [...prev, obj as PredictionResponse])
            }
          } catch { /* malformed line */ }
        }
      }
      setDone(true)
    } catch (err: any) {
      setError(err.message || "Unexpected error.")
    } finally {
      setLoading(false)
    }
  }

  const phishCount = results.filter(r => r.verdict === "phishing").length
  const safeCount  = results.filter(r => r.verdict === "legitimate").length

  return (
    <div className="space-y-5">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className="cursor-pointer rounded-xl border-2 border-dashed border-slate-600 bg-slate-800/30 px-6 py-10 text-center transition-colors hover:border-blue-500 hover:bg-slate-800/50"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={e => handleFile(e.target.files?.[0] ?? null)}
          />
          <UploadIcon className="mx-auto h-8 w-8 text-slate-500 mb-3" />
          {file ? (
            <div>
              <p className="font-semibold text-slate-200">{file.name}</p>
              <p className="text-xs text-slate-500 mt-1">
                {(file.size / 1024).toFixed(1)} KB — click to change
              </p>
            </div>
          ) : (
            <div>
              <p className="font-semibold text-slate-300">Drop CSV here or click to browse</p>
              <p className="text-xs text-slate-500 mt-1">
                One URL per row. First column or any column named <code>url/link/domain</code> is used.
              </p>
            </div>
          )}
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={loading || !file}
          className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-3.5 font-semibold text-white shadow-lg transition-all hover:from-blue-700 hover:to-purple-700 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <div className="flex items-center justify-center gap-3">
            {loading ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
                  className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white"
                />
                <span>
                  {total > 0
                    ? `Scanning ${results.length} / ${total}…`
                    : "Uploading…"}
                </span>
              </>
            ) : (
              <><UploadIcon className="h-5 w-5" /><span>Upload & Scan</span></>
            )}
          </div>
        </button>
      </form>

      {/* Progress */}
      {(loading || done) && total > 0 && (
        <div className="space-y-2">
          <div className="h-1.5 w-full rounded-full bg-slate-700 overflow-hidden">
            <motion.div
              animate={{ width: `${(results.length / total) * 100}%` }}
              transition={{ duration: 0.3 }}
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            />
          </div>
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>
              <ClockIcon className="inline h-3 w-3 mr-1" />
              {results.length} / {total} scanned
            </span>
            {done && (
              <span className="flex items-center gap-1 text-emerald-400">
                <CheckCircle2Icon className="h-3.5 w-3.5" /> Complete
              </span>
            )}
          </div>

          {done && results.length > 0 && (
            <div className="grid grid-cols-3 gap-3 pt-1">
              {[
                { label: "Phishing",   count: phishCount,                              color: "text-red-400"     },
                { label: "Legitimate", count: safeCount,                               color: "text-emerald-400" },
                { label: "Other",      count: results.length - phishCount - safeCount, color: "text-slate-400"   },
              ].map(s => (
                <div key={s.label} className="rounded-lg bg-slate-800/50 px-3 py-2 text-center">
                  <p className={`text-lg font-bold ${s.color}`}>{s.count}</p>
                  <p className="text-xs text-slate-500">{s.label}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Results stream */}
      {results.length > 0 && (
        <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
          <AnimatePresence initial={false}>
            {results.map((r, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.22 }}
              >
                <ResultCard result={r} />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}


export default function HomePage() {
  return (
    <Toast.Provider swipeDirection="right">
      <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        {/* Background glows */}
        <div className="pointer-events-none fixed inset-0">
          <div className="absolute -top-40 left-1/3 h-96 w-96 rounded-full bg-blue-600/10 blur-3xl" />
          <div className="absolute bottom-0 right-1/4 h-80 w-80 rounded-full bg-purple-600/10 blur-3xl" />
        </div>

        <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-16">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="w-full max-w-2xl space-y-6"
          >
            {/* Header */}
            <motion.div variants={itemVariants} className="space-y-2 text-center">
              <div className="flex items-center justify-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 p-2.5 shadow-lg shadow-blue-600/20">
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

            {/* Main card */}
            <motion.div
              variants={itemVariants}
              className="rounded-2xl border border-slate-800/50 bg-slate-900/60 backdrop-blur-xl shadow-2xl"
            >
              <Tabs.Root defaultValue="single" className="w-full">
                {/* Tab list */}
                <Tabs.List className="flex rounded-t-2xl overflow-hidden border-b border-slate-800/60">
                  {[
                    { value: "single", label: "Single URL", Icon: GlobeIcon },
                    { value: "bulk",   label: "Bulk URLs",  Icon: LinkIcon  },
                    // { value: "csv",    label: "CSV Upload", Icon: UploadIcon },
                  ].map(({ value, label, Icon }) => (
                    <Tabs.Trigger
                      key={value}
                      value={value}
                      className="flex-1 flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-semibold text-slate-500
                        transition-colors hover:text-slate-300
                        data-[state=active]:text-white data-[state=active]:bg-slate-800/60
                        data-[state=inactive]:bg-slate-900/40"
                    >
                      <Icon className="h-4 w-4" />
                      <span className="hidden sm:inline">{label}</span>
                    </Tabs.Trigger>
                  ))}
                </Tabs.List>

                <div className="p-7">
                  <Tabs.Content value="single"><SinglePanel /></Tabs.Content>
                  <Tabs.Content value="bulk"><BulkPanel /></Tabs.Content>
                  {/* <Tabs.Content value="csv"><CSVPanel /></Tabs.Content> */}
                </div>
              </Tabs.Root>
            </motion.div>

            {/* Stats footer */}
            <motion.div variants={itemVariants} className="grid grid-cols-3 gap-4">
              {[
                { label: "Layer 1", value: "TLS / URL",  sub: "threshold 0.25" },
                { label: "Layer 2", value: "HTML + URL", sub: "threshold 0.20" },
                { label: "AUC",     value: "0.9705",     sub: "combined score" },
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
      </div>
    </Toast.Provider>
  )
}