import { useEffect, useRef } from "react";
import { Alert, Badge, Button, Card, Group, Loader, ScrollArea, Stack, Text, Title } from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";
import type { SessionStats } from "../api/index.ts";
import type { PipelineStatus, TranscriptLine } from "../hooks/useAudioStream.ts";
import type { WebRTCMetrics } from "../transport/index.ts";

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function TranscriptBox({ label, lines }: { label: string; lines: TranscriptLine[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [lines.length]);

  return (
    <Card withBorder p="xs">
      <Title order={6} mb={4}>
        {label}
      </Title>
      <ScrollArea h={150} viewportRef={scrollRef} offsetScrollbars data-testid={`transcript-${label.toLowerCase().replace(/\s+/g, "-")}`}>
        <Stack gap={2}>
          {lines.map((line, i) => (
            <Group key={i} gap="xs" wrap="nowrap" align="flex-start">
              <Text size="xs" c="dimmed" style={{ whiteSpace: "nowrap", flexShrink: 0 }}>
                {formatTime(line.timestamp)}
              </Text>
              <Text size="sm">{line.text}</Text>
            </Group>
          ))}
        </Stack>
      </ScrollArea>
    </Card>
  );
}

function phaseBadge(status: PipelineStatus): { color: string; label: string } {
  switch (status.phase) {
    case "connecting": return { color: "yellow", label: "Connecting" };
    case "loading": return { color: "blue", label: "Loading Models" };
    case "ready": return { color: "green", label: "Ready" };
    case "streaming": return { color: "green", label: "Streaming" };
    case "error": return { color: "red", label: "Error" };
    case "stopped": return { color: "gray", label: "Stopped" };
    default: return { color: "gray", label: status.phase };
  }
}

function rtcStateBadge(state: string): { color: string } {
  switch (state) {
    case "connected": return { color: "green" };
    case "connecting": case "new": return { color: "yellow" };
    case "disconnected": case "failed": case "closed": return { color: "red" };
    default: return { color: "gray" };
  }
}

export function ActiveSessionPanel({
  sessionId,
  pipelineId,
  connected,
  liveStats,
  transcripts,
  streamLabels,
  rtcMetrics,
  pipelineStatus,
  onStop,
}: {
  sessionId: string;
  pipelineId: string;
  connected: boolean;
  liveStats: SessionStats | null;
  transcripts: Record<string, TranscriptLine[]>;
  streamLabels: Record<string, string>;
  rtcMetrics: WebRTCMetrics | null;
  pipelineStatus: PipelineStatus;
  onStop: () => void;
}) {
  const streamNames = Object.keys(transcripts);
  const phase = phaseBadge(pipelineStatus);
  const isLoading = pipelineStatus.phase === "connecting" || pipelineStatus.phase === "loading";

  return (
    <Card withBorder p="md">
      <Group justify="space-between" mb="sm">
        <Text fw={600} size="lg">
          Active Session
        </Text>
        <Group gap="xs">
          {rtcMetrics && (
            <Badge
              color={rtcStateBadge(rtcMetrics.connectionState).color}
              variant="dot"
              size="sm"
              title={`ICE: ${rtcMetrics.iceState}`}
            >
              RTC {rtcMetrics.connectionState}
            </Badge>
          )}
          <Badge color={phase.color} variant={isLoading ? "outline" : "filled"} size="sm">
            {isLoading && <Loader size={10} color={phase.color} mr={4} />}
            {phase.label}
          </Badge>
        </Group>
      </Group>

      {pipelineStatus.phase === "error" && (
        <Alert icon={<IconAlertTriangle size={16} />} color="red" mb="sm" title="Pipeline Error">
          {pipelineStatus.detail}
        </Alert>
      )}

      {isLoading && pipelineStatus.detail && (
        <Text size="sm" c="dimmed" mb="sm">{pipelineStatus.detail}</Text>
      )}

      <Stack gap="xs">
        <Text size="sm">Session: {sessionId}</Text>
        <Text size="sm">Pipeline: {pipelineId}</Text>

        {liveStats && (
          <>
            <Text size="sm">
              Duration: {liveStats.duration_seconds.toFixed(0)}s
            </Text>
            <Text size="sm">
              Audio In: {bytes(liveStats.bytes_received)} ({liveStats.chunks_received} chunks)
            </Text>
            <Text size="sm">
              Audio Out: {bytes(liveStats.bytes_sent)} ({liveStats.chunks_sent} chunks)
            </Text>
            <Text size="sm" fw={600} c={liveStats.audio_delay_seconds > 10 ? "red" : liveStats.audio_delay_seconds > 5 ? "yellow" : undefined}>
              Audio Delay: {liveStats.audio_delay_seconds.toFixed(2)}s
            </Text>
            {(liveStats.pending_sentences > 0 || liveStats.queued_audio_seconds > 0) && (
              <Group gap="md">
                <Text size="xs" c="dimmed">
                  Pending: {liveStats.pending_sentences} sentences
                </Text>
                <Text size="xs" c="dimmed">
                  Queued: {liveStats.queued_audio_seconds.toFixed(1)}s audio
                </Text>
              </Group>
            )}
          </>
        )}

        {rtcMetrics && connected && (
          <Group gap="md">
            {rtcMetrics.roundTripMs != null && (
              <Text size="xs" c="dimmed">RTT: {rtcMetrics.roundTripMs}ms</Text>
            )}
            {rtcMetrics.jitterMs != null && (
              <Text size="xs" c="dimmed">Jitter: {rtcMetrics.jitterMs}ms</Text>
            )}
            {rtcMetrics.packetsLost > 0 && (
              <Text size="xs" c="red">Lost: {rtcMetrics.packetsLost}</Text>
            )}
          </Group>
        )}

        {streamNames.length > 0 && (
          <Stack gap="xs" mt="xs">
            {streamNames.map((name) => (
              <TranscriptBox
                key={name}
                label={streamLabels[name] || name}
                lines={transcripts[name]}
              />
            ))}
          </Stack>
        )}

        <Group justify="flex-end" mt="xs">
          <Button color="red" variant="outline" onClick={onStop}>
            Stop
          </Button>
        </Group>
      </Stack>
    </Card>
  );
}
