import { useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  FileButton,
  Group,
  Modal,
  NativeSelect,
  NumberInput,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
} from "@mantine/core";
import { createSession, fetchPipelines, fetchStages } from "../api/index.ts";
import type { PipelineInfo, Session, StageInfo, StageKind } from "../api/index.ts";
import { useAudioDevices } from "../hooks/useAudioDevices.ts";

export type AudioSourceType = "mic" | "file";

export interface AudioSource {
  type: AudioSourceType;
  file?: File;
}

const COMPOSED_PIPELINE_ID = "composed";
const STAGE_KINDS: StageKind[] = ["listen", "translate", "speak", "prosody"];

function defaultStageId(stages: StageInfo[], kind: StageKind): string {
  const matching = stages.filter((s) => s.kind === kind);
  const preferred = matching.find((s) => s.default_for_kind) ?? matching[0];
  return preferred?.id ?? "";
}

function stageOptions(stages: StageInfo[], kind: StageKind, allowNone = false) {
  const options = stages
    .filter((s) => s.kind === kind)
    .map((s) => ({ value: s.id, label: s.name }));
  if (allowNone) {
    return [{ value: "", label: "None" }, ...options];
  }
  return options;
}

export function NewSessionModal({
  opened,
  onClose,
  onCreated,
}: {
  opened: boolean;
  onClose: () => void;
  onCreated: (
    session: Session,
    pipelineInfo: PipelineInfo,
    inputDeviceId: string,
    outputDeviceId: string,
    audioSource: AudioSource,
  ) => void;
}) {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [stages, setStages] = useState<StageInfo[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState("");
  const [listenStage, setListenStage] = useState("");
  const [translateStage, setTranslateStage] = useState("");
  const [speakStage, setSpeakStage] = useState("");
  const [prosodyStage, setProsodyStage] = useState("");
  const [label, setLabel] = useState("");
  const [inputDevice, setInputDevice] = useState("");
  const [outputDevice, setOutputDevice] = useState("");
  const [creating, setCreating] = useState(false);
  const [sourceType, setSourceType] = useState<AudioSourceType>("mic");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioContextSeconds, setAudioContextSeconds] = useState<number>(0);
  const resetRef = useRef<() => void>(null);
  const { inputs, outputs } = useAudioDevices();

  useEffect(() => {
    if (!opened) return;
    void Promise.all([fetchPipelines(), fetchStages()]).then(([p, s]) => {
      setPipelines(p);
      setStages(s);
      if (p.length > 0) setSelectedPipeline(p[0].id);
      setListenStage(defaultStageId(s, "listen"));
      setTranslateStage(defaultStageId(s, "translate"));
      setSpeakStage(defaultStageId(s, "speak"));
      setProsodyStage(defaultStageId(s, "prosody"));
    });
  }, [opened]);

  useEffect(() => {
    if (inputs.length > 0 && !inputDevice) setInputDevice(inputs[0].deviceId);
  }, [inputs, inputDevice]);

  useEffect(() => {
    if (outputs.length > 0 && !outputDevice) setOutputDevice(outputs[0].deviceId);
  }, [outputs, outputDevice]);

  const selected = pipelines.find((p) => p.id === selectedPipeline);
  const isComposed = selectedPipeline === COMPOSED_PIPELINE_ID;

  const composedReady = useMemo(() => {
    if (!isComposed) return true;
    return Boolean(listenStage && translateStage && speakStage);
  }, [isComposed, listenStage, translateStage, speakStage]);

  const handleCreate = async () => {
    if (!selectedPipeline || !selected) return;
    setCreating(true);
    try {
      const session = await createSession({
        pipeline_id: selectedPipeline,
        label: label || undefined,
        audio_context_seconds: audioContextSeconds || undefined,
        stages: isComposed
          ? {
              listen: listenStage,
              translate: translateStage,
              speak: speakStage,
              prosody: prosodyStage || null,
            }
          : undefined,
      });
      const source: AudioSource =
        sourceType === "file" && selectedFile
          ? { type: "file", file: selectedFile }
          : { type: "mic" };
      setLabel("");
      setSelectedFile(null);
      resetRef.current?.();
      onCreated(session, selected, inputDevice, outputDevice, source);
      onClose();
    } finally {
      setCreating(false);
    }
  };

  return (
    <Modal opened={opened} onClose={onClose} title="New Session" centered>
      <Stack gap="sm">
        <NativeSelect
          label="Pipeline"
          data={pipelines.map((p) => ({ value: p.id, label: p.name }))}
          value={selectedPipeline}
          onChange={(e) => setSelectedPipeline(e.currentTarget.value)}
        />
        {selected && (
          <Text size="xs" c="dimmed">
            {selected.description}
          </Text>
        )}
        {isComposed && (
          <>
            {STAGE_KINDS.map((kind) => {
              const allowNone = kind === "prosody";
              const value =
                kind === "listen"
                  ? listenStage
                  : kind === "translate"
                    ? translateStage
                    : kind === "speak"
                      ? speakStage
                      : prosodyStage;
              const onChange =
                kind === "listen"
                  ? setListenStage
                  : kind === "translate"
                    ? setTranslateStage
                    : kind === "speak"
                      ? setSpeakStage
                      : setProsodyStage;
              const labelText =
                kind === "listen"
                  ? "Listen stage"
                  : kind === "translate"
                    ? "Translate stage"
                    : kind === "speak"
                      ? "Speak stage"
                      : "Prosody stage";
              return (
                <NativeSelect
                  key={kind}
                  label={labelText}
                  data={stageOptions(stages, kind, allowNone)}
                  value={value}
                  onChange={(e) => onChange(e.currentTarget.value)}
                />
              );
            })}
          </>
        )}
        <TextInput
          label="Label (optional)"
          placeholder="My session"
          value={label}
          onChange={(e) => setLabel(e.currentTarget.value)}
        />
        <SegmentedControl
          fullWidth
          data={[
            { value: "mic", label: "Live Microphone" },
            { value: "file", label: "MP3 File" },
          ]}
          value={sourceType}
          onChange={(v) => setSourceType(v as AudioSourceType)}
        />
        {sourceType === "mic" ? (
          <NativeSelect
            label="Audio Input"
            data={
              inputs.length > 0
                ? inputs.map((d) => ({ value: d.deviceId, label: d.label }))
                : [{ value: "", label: "No devices found" }]
            }
            value={inputDevice}
            onChange={(e) => setInputDevice(e.currentTarget.value)}
          />
        ) : (
          <Group gap="sm">
            <FileButton
              resetRef={resetRef}
              onChange={(f) => setSelectedFile(f)}
              accept="audio/mpeg,audio/mp3,.mp3"
            >
              {(props) => (
                <Button variant="light" {...props}>
                  Choose MP3 file
                </Button>
              )}
            </FileButton>
            <Text size="sm" c="dimmed" style={{ flex: 1 }}>
              {selectedFile ? selectedFile.name : "No file selected"}
            </Text>
          </Group>
        )}
        <NativeSelect
          label="Audio Output"
          data={
            outputs.length > 0
              ? outputs.map((d) => ({ value: d.deviceId, label: d.label }))
              : [{ value: "", label: "No devices found" }]
          }
          value={outputDevice}
          onChange={(e) => setOutputDevice(e.currentTarget.value)}
        />
        <NumberInput
          label="Audio Context (seconds)"
          description="Previous audio fed to the model for continuity (0 = none)"
          value={audioContextSeconds}
          onChange={(v) => setAudioContextSeconds(typeof v === "number" ? v : 0)}
          min={0}
          max={30}
          step={1}
          allowDecimal={false}
        />
        <Group justify="flex-end" mt="xs">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            loading={creating}
            disabled={
              !selectedPipeline ||
              !composedReady ||
              (sourceType === "file" && !selectedFile)
            }
          >
            Start Session
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
