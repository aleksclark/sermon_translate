import { useMemo, useState } from "react";
import {
  AppShell,
  Grid,
  MantineProvider,
  Stack,
  Title,
  ActionIcon,
  Group,
  useMantineColorScheme,
  createTheme,
} from "@mantine/core";
import { IconMoon, IconSun } from "@tabler/icons-react";
import type { Session } from "./api/index.ts";
import type { AudioSource } from "./components/NewSessionModal.tsx";
import { updateSession } from "./api/index.ts";
import { ServerStatsPanel } from "./components/ServerStatsPanel.tsx";
import { SessionListPanel } from "./components/SessionListPanel.tsx";
import { NewSessionModal } from "./components/NewSessionModal.tsx";
import { ActiveSessionPanel } from "./components/ActiveSessionPanel.tsx";
import { useServerStats } from "./hooks/useServerStats.ts";
import { useSessions } from "./hooks/useSessions.ts";
import { useAudioStream } from "./hooks/useAudioStream.ts";

const theme = createTheme({
  primaryColor: "blue",
  fontFamily: "Inter, system-ui, -apple-system, sans-serif",
});

interface ActiveSession {
  session: Session;
  inputDeviceId: string;
  outputDeviceId: string;
  audioSource: AudioSource;
}

function ColorSchemeToggle() {
  const { colorScheme, toggleColorScheme } = useMantineColorScheme();
  return (
    <ActionIcon
      variant="subtle"
      size="lg"
      onClick={() => toggleColorScheme()}
      aria-label="Toggle color scheme"
    >
      {colorScheme === "dark" ? <IconSun size={20} /> : <IconMoon size={20} />}
    </ActionIcon>
  );
}

function AppContent() {
  const stats = useServerStats();
  const { sessions, refresh } = useSessions();
  const [activeSession, setActiveSession] = useState<ActiveSession | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [newSessionOpen, setNewSessionOpen] = useState(false);

  const streamOptions = useMemo(() => {
    if (!activeSession) return null;
    return {
      sessionId: activeSession.session.id,
      sampleRate: activeSession.session.sample_rate,
      channels: activeSession.session.channels,
      inputDeviceId: activeSession.inputDeviceId,
      outputDeviceId: activeSession.outputDeviceId,
      audioSource: activeSession.audioSource,
    };
  }, [activeSession]);

  const { connected, liveStats, transcript, stop } = useAudioStream(streamOptions);

  const handleCreated = (
    session: Session,
    inputDeviceId: string,
    outputDeviceId: string,
    audioSource: AudioSource,
  ) => {
    setActiveSession({ session, inputDeviceId, outputDeviceId, audioSource });
    setSelectedId(session.id);
    refresh();
  };

  const handleStop = async () => {
    const id = activeSession?.session.id;
    stop();
    setActiveSession(null);
    if (id) {
      try {
        await updateSession(id, { status: "closed" });
      } catch {
        // server may have already closed it
      }
    }
    refresh();
  };

  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header p="xs" px="md">
        <Group justify="space-between" h="100%">
          <Title order={3}>Sermon Translate</Title>
          <ColorSchemeToggle />
        </Group>
      </AppShell.Header>

      <AppShell.Main>
        <Grid>
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Stack gap="md">
              <ServerStatsPanel stats={stats} />
              <SessionListPanel
                sessions={sessions}
                activeId={selectedId}
                onSelect={setSelectedId}
                onNew={() => setNewSessionOpen(true)}
              />
            </Stack>
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 8 }}>
            <Stack gap="md">
              {activeSession && (
                <ActiveSessionPanel
                  sessionId={activeSession.session.id}
                  pipelineId={activeSession.session.pipeline_id}
                  connected={connected}
                  liveStats={liveStats}
                  transcript={transcript}
                  onStop={handleStop}
                />
              )}
            </Stack>
          </Grid.Col>
        </Grid>
      </AppShell.Main>

      <NewSessionModal
        opened={newSessionOpen}
        onClose={() => setNewSessionOpen(false)}
        onCreated={handleCreated}
      />
    </AppShell>
  );
}

export default function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <AppContent />
    </MantineProvider>
  );
}
