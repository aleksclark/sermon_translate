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
import type { PipelineInfo, Session } from "./api/index.ts";
import type { AudioSource } from "./components/NewSessionModal.tsx";
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
  pipelineInfo: PipelineInfo;
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

  const { connected, muted, liveStats, transcripts, stop, toggleMute } = useAudioStream(streamOptions);

  const streamLabels = useMemo(() => {
    if (!activeSession) return {};
    const labels: Record<string, string> = {};
    for (const s of activeSession.pipelineInfo.output_streams) {
      if (s.kind === "text" && s.label) {
        labels[s.name] = s.label;
      }
    }
    return labels;
  }, [activeSession]);

  const handleCreated = (
    session: Session,
    pipelineInfo: PipelineInfo,
    inputDeviceId: string,
    outputDeviceId: string,
    audioSource: AudioSource,
  ) => {
    setActiveSession({ session, pipelineInfo, inputDeviceId, outputDeviceId, audioSource });
    setSelectedId(session.id);
    refresh();
  };

  const handleStop = async () => {
    stop();
    setActiveSession(null);
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
                  muted={muted}
                  liveStats={liveStats}
                  transcripts={transcripts}
                  streamLabels={streamLabels}
                  onStop={handleStop}
                  onToggleMute={toggleMute}
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
