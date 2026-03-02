import { ActionIcon, Badge, Card, Group, Stack, Text } from "@mantine/core";
import { IconPlus } from "@tabler/icons-react";
import type { Session } from "../api/index.ts";

const STATUS_COLOR: Record<string, string> = {
  created: "blue",
  active: "green",
  paused: "yellow",
  closed: "gray",
};

export function SessionListPanel({
  sessions,
  activeId,
  onSelect,
  onNew,
}: {
  sessions: Session[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <Card withBorder p="md">
      <Group justify="space-between" mb="sm">
        <Text fw={600} size="lg">
          Sessions
        </Text>
        <ActionIcon variant="subtle" size="lg" onClick={onNew} aria-label="New session">
          <IconPlus size={20} />
        </ActionIcon>
      </Group>
      {sessions.length === 0 ? (
        <Text c="dimmed" size="sm">
          No sessions yet.
        </Text>
      ) : (
        <Stack gap="xs">
          {sessions.map((s) => (
            <Card
              key={s.id}
              withBorder
              p="xs"
              style={{
                cursor: "pointer",
                outline: s.id === activeId ? "2px solid var(--mantine-color-blue-5)" : undefined,
              }}
              onClick={() => onSelect(s.id)}
            >
              <Group justify="space-between">
                <Text size="sm" fw={500}>
                  {s.label || s.id}
                </Text>
                <Badge color={STATUS_COLOR[s.status] ?? "gray"} size="sm">
                  {s.status}
                </Badge>
              </Group>
              <Text size="xs" c="dimmed">
                Pipeline: {s.pipeline_id} &middot; {s.sample_rate / 1000}kHz
              </Text>
            </Card>
          ))}
        </Stack>
      )}
    </Card>
  );
}
