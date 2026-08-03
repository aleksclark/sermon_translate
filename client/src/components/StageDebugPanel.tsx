import { useEffect, useRef } from "react";
import { Badge, Card, Collapse, Group, ScrollArea, Stack, Text, Title } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import type {
  ListenProduct,
  ProsodyToken,
  StageSelection,
  TranslateProduct,
  WordSpan,
} from "../api/index.ts";
import type { MetadataUpdate, StageProductUpdate } from "../hooks/useAudioStream.ts";

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function isListenProduct(product: unknown): product is ListenProduct {
  return (
    typeof product === "object" &&
    product !== null &&
    "utterance_id" in product &&
    "text" in product &&
    "words" in product
  );
}

function isTranslateProduct(product: unknown): product is TranslateProduct {
  return (
    typeof product === "object" &&
    product !== null &&
    "source_utterance_id" in product &&
    "target_utterance_id" in product &&
    "text" in product
  );
}

function ProsodyChip({ token }: { token: ProsodyToken }) {
  return (
    <Badge size="xs" variant="light" color="grape">
      p{token.pitch_median}/e{token.energy}
      {token.f0_hz != null ? ` ${token.f0_hz.toFixed(0)}Hz` : ""}
    </Badge>
  );
}

function WordRow({ word }: { word: WordSpan }) {
  return (
    <Group gap={4} wrap="wrap">
      <Text size="sm">{word.text}</Text>
      {word.prosody && <ProsodyChip token={word.prosody} />}
    </Group>
  );
}

function DebugSection({
  title,
  testId,
  defaultOpen = true,
  children,
}: {
  title: string;
  testId: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [opened, { toggle }] = useDisclosure(defaultOpen);
  return (
    <Card withBorder p="xs" data-testid={testId}>
      <Group justify="space-between" mb={opened ? 4 : 0} style={{ cursor: "pointer" }} onClick={toggle}>
        <Title order={6}>{title}</Title>
        <Text size="xs" c="dimmed">
          {opened ? "Hide" : "Show"}
        </Text>
      </Group>
      <Collapse in={opened}>{children}</Collapse>
    </Card>
  );
}

function ProductList({
  updates,
  render,
}: {
  updates: StageProductUpdate[];
  render: (update: StageProductUpdate) => React.ReactNode;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [updates.length]);

  if (updates.length === 0) {
    return (
      <Text size="xs" c="dimmed">
        No products yet
      </Text>
    );
  }

  return (
    <ScrollArea h={160} viewportRef={scrollRef} offsetScrollbars>
      <Stack gap={6}>
        {updates.map((update, i) => (
          <Stack key={i} gap={2}>
            <Text size="xs" c="dimmed">
              {formatTime(update.timestamp)}
            </Text>
            {render(update)}
          </Stack>
        ))}
      </Stack>
    </ScrollArea>
  );
}

function ProsodyTable({ frames }: { frames: MetadataUpdate[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const recent = frames.slice(-40);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [frames.length]);

  if (recent.length === 0) {
    return (
      <Text size="xs" c="dimmed">
        No prosody frames yet
      </Text>
    );
  }

  return (
    <ScrollArea h={140} viewportRef={scrollRef} offsetScrollbars>
      <Stack gap={2}>
        {recent.map((frame, i) => {
          const p = frame.envelope.prosody;
          return (
            <Group key={i} gap="xs" wrap="nowrap">
              <Text size="xs" c="dimmed" w={70}>
                #{frame.envelope.sequence}
              </Text>
              <Text size="xs" w={70}>
                f0 {p?.f0_hz != null ? p.f0_hz.toFixed(0) : "—"}
              </Text>
              <Text size="xs" w={70}>
                e {p?.energy != null ? p.energy.toFixed(2) : "—"}
              </Text>
              <Badge size="xs" color={p?.is_pause ? "gray" : "teal"} variant="light">
                {p?.is_pause ? "pause" : "speech"}
              </Badge>
            </Group>
          );
        })}
      </Stack>
    </ScrollArea>
  );
}

export function StageDebugPanel({
  stageProducts,
  metadata,
  stages,
  liveStats,
}: {
  stageProducts: Record<string, StageProductUpdate[]>;
  metadata: Record<string, MetadataUpdate[]>;
  stages: StageSelection | null;
  liveStats: { bytes_sent: number; chunks_sent: number } | null;
}) {
  const listen = stageProducts.listen || [];
  const translate = stageProducts.translate || [];
  const speak = stageProducts.speak || [];
  const prosodyFrames = metadata.prosody || [];
  const lastTranslate = translate[translate.length - 1];
  const lastTargetId =
    lastTranslate && isTranslateProduct(lastTranslate.product)
      ? lastTranslate.product.target_utterance_id
      : null;

  return (
    <Stack gap="xs" mt="xs" data-testid="stage-debug-panel">
      {stages && (
        <Text size="xs" c="dimmed" data-testid="stage-selection-summary">
          listen={stages.listen} · translate={stages.translate} · speak={stages.speak}
          {stages.prosody ? ` · prosody=${stages.prosody}` : ""}
        </Text>
      )}

      <DebugSection title="Listen products" testId="stage-debug-listen" defaultOpen>
        <ProductList
          updates={listen}
          render={(update) => {
            if (!isListenProduct(update.product)) {
              return <Text size="sm">{JSON.stringify(update.product)}</Text>;
            }
            return (
              <Stack gap={2}>
                <Text size="sm">{update.product.text}</Text>
                <Group gap={6} wrap="wrap">
                  {update.product.words.map((word, i) => (
                    <WordRow key={i} word={word} />
                  ))}
                </Group>
              </Stack>
            );
          }}
        />
      </DebugSection>

      <DebugSection title="Translate products" testId="stage-debug-translate" defaultOpen>
        <ProductList
          updates={translate}
          render={(update) => {
            if (!isTranslateProduct(update.product)) {
              return <Text size="sm">{JSON.stringify(update.product)}</Text>;
            }
            const markers = update.product.instructions?.markers ?? [];
            return (
              <Stack gap={2}>
                <Text size="sm">{update.product.text}</Text>
                <Text size="xs" c="dimmed">
                  markers: {markers.length}
                  {markers[0] && typeof markers[0].word === "string"
                    ? ` (e.g. ${String(markers[0].word)})`
                    : ""}
                </Text>
                <Group gap={6} wrap="wrap">
                  {update.product.words.map((word, i) => (
                    <WordRow key={i} word={word} />
                  ))}
                </Group>
              </Stack>
            );
          }}
        />
      </DebugSection>

      <DebugSection title="Prosody frames" testId="stage-debug-prosody" defaultOpen={false}>
        <ProsodyTable frames={prosodyFrames} />
      </DebugSection>

      <DebugSection title="Speak status" testId="stage-debug-speak" defaultOpen={false}>
        <Stack gap={2}>
          <Text size="sm">
            Last target utterance: {lastTargetId ?? "—"}
          </Text>
          <Text size="sm">
            Audio out: {liveStats ? `${liveStats.chunks_sent} chunks / ${liveStats.bytes_sent} B` : "—"}
          </Text>
          {speak.length > 0 && (
            <Text size="xs" c="dimmed">
              Speak products received: {speak.length}
            </Text>
          )}
        </Stack>
      </DebugSection>
    </Stack>
  );
}
