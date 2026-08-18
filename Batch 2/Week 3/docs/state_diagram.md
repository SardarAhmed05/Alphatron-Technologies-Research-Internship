# System State Transition Diagram

This state diagram depicts the state transitions of the Conversational RAG Chatbot application from initialization to document ingestion, query execution, vector store resetting, and memory management.

```mermaid
stateDiagram-v2
    [*] --> Idle: Application Start

    state Idle {
        [*] --> WaitingForUser
        WaitingForUser --> DocumentSelected: User Selects File
        WaitingForUser --> QuerySubmitted: User Enters Question
        WaitingForUser --> ClearRequested: Reset DB / Memory
    }

    state IngestingDocument {
        [*] --> ReadingFile
        ReadingFile --> ValidatingFormat
        ValidatingFormat --> ParsingContent
        ParsingContent --> ChunkingText
        ChunkingText --> GeneratingEmbeddings
        GeneratingEmbeddings --> WritingToChroma
        WritingToChroma --> IngestionComplete
    }

    state ProcessingQuery {
        [*] --> CheckingHistory
        CheckingHistory --> RephrasingQuery: History Present
        CheckingHistory --> VectorRetrieval: No History
        RephrasingQuery --> VectorRetrieval
        VectorRetrieval --> FormattingContext
        FormattingContext --> InvokingLLM
        InvokingLLM --> UpdatingMemory
        UpdatingMemory --> QueryComplete
    }

    state ResettingState {
        [*] --> ClearingChromaDB
        ClearingChromaDB --> ResettingMemory
        ResettingMemory --> StateResetComplete
    }

    DocumentSelected --> IngestingDocument
    IngestionComplete --> Idle: Update Collection Stats

    QuerySubmitted --> ProcessingQuery
    QueryComplete --> Idle: Render Answer & Citations

    ClearRequested --> ResettingState
    StateResetComplete --> Idle: Ready
```
