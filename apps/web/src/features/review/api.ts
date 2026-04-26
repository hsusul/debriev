import type { ClaimReviewHistoryApiPayload, ReviewApiPayload } from "@/features/review/adapters"
import type { ResolutionSubmissionRequest } from "@/features/review/types"

interface ClaimDecisionMutationResponse {
  decision: {
    id: string
    claim_unit_id: string
  }
  claim_review_state: {
    removed_from_active_queue: boolean
  }
  draft_queue: {
    next_claim_id: string | null
  }
}

export function resolveDraftId() {
  const fromQuery = new URLSearchParams(window.location.search).get("draftId")
  if (fromQuery && fromQuery.trim().length > 0) {
    return fromQuery.trim()
  }

  const fromEnv = import.meta.env.VITE_DEBRIEV_DRAFT_ID
  if (fromEnv && fromEnv.trim().length > 0) {
    return fromEnv.trim()
  }

  return null
}

export async function fetchDraftReviewState(draftId: string): Promise<ReviewApiPayload> {
  return requestJson<ReviewApiPayload>(`/api/v1/drafts/${draftId}/review-state`)
}

export async function rerunDraftReview(draftId: string): Promise<ReviewApiPayload> {
  return requestJson<ReviewApiPayload>(`/api/v1/drafts/${draftId}/review`, {
    method: "POST",
  })
}

export async function fetchClaimReviewHistory(claimId: string): Promise<ClaimReviewHistoryApiPayload> {
  return requestJson<ClaimReviewHistoryApiPayload>(`/api/v1/claims/${claimId}/review-history`)
}

export async function submitClaimDecision(
  payload: ResolutionSubmissionRequest,
): Promise<ClaimDecisionMutationResponse> {
  return requestJson<ClaimDecisionMutationResponse>(`/api/v1/claims/${payload.claimId}/decisions`, {
    method: "POST",
    body: JSON.stringify({
      action: payload.action,
      note: payload.note,
      proposed_replacement_text: payload.proposedReplacementText,
    }),
  })
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${resolveApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    throw new Error(await buildApiErrorMessage(response))
  }

  return (await response.json()) as T
}

async function buildApiErrorMessage(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string }
    if (payload.detail) {
      return payload.detail
    }
  } catch {
    // ignore invalid json payloads and fall back to status text
  }

  return `Request failed (${response.status})`
}

function resolveApiBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (!configuredBaseUrl) {
    return "http://localhost:8000"
  }

  return configuredBaseUrl.endsWith("/")
    ? configuredBaseUrl.slice(0, -1)
    : configuredBaseUrl
}
