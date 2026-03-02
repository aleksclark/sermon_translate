import { useEffect, useRef } from "react";
import { Badge, Button, Card, Group, ScrollArea, Stack, Text, Title } from "@mantine/core";
import type { SessionStats } from "../api/index.ts";
import type { TranscriptLine } from "../hooks/useAudioStream.ts";

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

export function ActiveSessionPanel({
  sessionId,
  pipelineId,
  connected,
  liveStats,
  transcripts,
  streamLabels,
  onStop,
}: {
  sessionId: string;
  pipelineId: string;
  connected: boolean;
  liveStats: SessionStats | null;
  transcripts: Record<string, TranscriptLine[]>;
  streamLabels: Record<string, string>;
  onStop: () => void;
}) {
  const streamNames = Object.keys(transcripts);

  return (
    <Card withBorder p="md">
      <Group justify="space-between" mb="sm">
        <Text fw={600} size="lg">
          Active Session
        </Text>
        <Badge color={connected ? "green" : "gray"}>
          {connected ? "Streaming" : "Disconnected"}
        </Badge>
      </Group>
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
            <Text size="sm">
              Pipeline Latency: {liveStats.pipeline_latency_ms.toFixed(0)}ms
            </Text>
            {liveStats.audio_delay_seconds > 0 && (
              <Text size="sm" c={liveStats.audio_delay_seconds > 5 ? "red" : liveStats.audio_delay_seconds > 2 ? "yellow" : undefined}>
                Audio Delay: {liveStats.audio_delay_seconds.toFixed(1)}s
              </Text>
            )}
          </>
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
