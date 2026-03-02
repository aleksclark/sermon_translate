import { Card, Group, Stack, Text } from "@mantine/core";
import {
  IconActivity,
  IconClock,
  IconDatabase,
  IconServer,
} from "@tabler/icons-react";
import type { ServerStats } from "../api/index.ts";

function fmt(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h}h ${m}m ${s}s`;
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function ServerStatsPanel({ stats }: { stats: ServerStats | null }) {
  if (!stats) {
    return (
      <Card withBorder p="md">
        <Text c="dimmed">Connecting to server...</Text>
      </Card>
    );
  }

  return (
    <Card withBorder p="md">
      <Text fw={600} size="lg" mb="sm">
        Server
      </Text>
      <Stack gap="xs">
        <Group gap="xs">
          <IconClock size={16} />
          <Text size="sm">Uptime: {fmt(stats.uptime_seconds)}</Text>
        </Group>
        <Group gap="xs">
          <IconActivity size={16} />
          <Text size="sm">
            Sessions: {stats.active_sessions} active / {stats.total_sessions}{" "}
            total
          </Text>
        </Group>
        <Group gap="xs">
          <IconDatabase size={16} />
          <Text size="sm">
            Processed: {bytes(stats.total_bytes_processed)}
          </Text>
        </Group>
        <Group gap="xs">
          <IconServer size={16} />
          <Text size="sm">Pipelines: {stats.available_pipelines}</Text>
        </Group>
      </Stack>
    </Card>
  );
}
