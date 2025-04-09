import { Conversation, Message } from '@/types/chat';
import { FolderInterface } from '@/types/folder';
import { t } from 'i18next';

export interface HomeInitialState {
  loading: boolean;
  lightMode: 'light' | 'dark';
  messageIsStreaming: boolean;
  folders: FolderInterface[];
  conversations: Conversation[];
  selectedConversation: Conversation | undefined;
  currentMessage: Message | undefined;
  showChatbar: boolean;
  currentFolder: FolderInterface | undefined;
  messageError: boolean;
  searchTerm: string;
  chatHistory: boolean;
  chatCompletionURL?: string;
  webSocketMode?: boolean;
  webSocketConnected?: boolean;
  webSocketURL?: string;
  webSocketSchema?: string;
  webSocketSchemas?: string[];
  enableIntermediateSteps?: boolean;
  expandIntermediateSteps?: boolean;
  intermediateStepOverride?: boolean;
  autoScroll?: boolean;
  additionalConfig: any;
}

export const initialState: HomeInitialState = {
  loading: false,
  lightMode: 'light',
  messageIsStreaming: false,
  folders: [],
  conversations: [],
  selectedConversation: undefined,
  currentMessage: undefined,
  showChatbar: true,
  currentFolder: undefined,
  messageError: false,
  searchTerm: '',
  chatHistory: process?.env?.NEXT_PUBLIC_CHAT_HISTORY_DEFAULT_ON === 'true' || false,
  chatCompletionURL: process?.env?.NEXT_PUBLIC_HTTP_CHAT_COMPLETION_URL || 'http://127.0.0.1:8000/chat/stream',
  webSocketMode: process?.env?.NEXT_PUBLIC_WEB_SOCKET_DEFAULT_ON === 'true' || false,
  webSocketConnected: false,
  webSocketURL: process?.env?.NEXT_PUBLIC_WEBSOCKET_CHAT_COMPLETION_URL || 'ws://127.0.0.1:8000/websocket',
  webSocketSchema: 'chat_stream',
  webSocketSchemas: ['chat_stream', 'chat', 'generate_stream', 'generate'],
  enableIntermediateSteps: process?.env?.NEXT_PUBLIC_ENABLE_INTERMEDIATE_STEPS ? process.env.NEXT_PUBLIC_ENABLE_INTERMEDIATE_STEPS === 'true' : true,
  expandIntermediateSteps: false,
  intermediateStepOverride: true,
  autoScroll: true,
  additionalConfig: {},
};
