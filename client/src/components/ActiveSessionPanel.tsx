import { useEffect, useRef } from "react";
import { Badge, Button, Card, Group, ScrollArea, Stack, Text } from "@mantine/core";
import type { SessionStats } from "../api/index.ts";

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function ActiveSessionPanel({
  sessionId,
  pipelineId,
  connected,
  liveStats,
  transcript,
  onStop,
}: {
  sessionId: string;
  pipelineId: string;
  connected: boolean;
  liveStats: SessionStats | null;
  transcript: string[];
  onStop: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [transcript.length]);

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
          </>
        )}

        {transcript.length > 0 && (
          <>
            <Text fw={500} size="sm" mt="xs">
              Transcript
            </Text>
            <ScrollArea h={200} viewportRef={scrollRef} offsetScrollbars data-testid="transcript-area">
              <Stack gap={4}>
                {transcript.map((line, i) => (
                  <Text key={i} size="sm" c="dimmed">
                    {line}
                  </Text>
                ))}
              </Stack>
            </ScrollArea>
          </>
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
