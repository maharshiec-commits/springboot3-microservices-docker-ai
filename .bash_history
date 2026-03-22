import java.util.concurrent.TimeUnit;

@Slf4j
@Configuration
public class WebClientConfig {

    @Bean
    public WebClient llmWebClient(LlmProperties props) {
        HttpClient httpClient = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, props.getTimeoutSeconds() * 1000)
                .responseTimeout(Duration.ofSeconds(props.getTimeoutSeconds()))
                .doOnConnected(conn -> conn
                        .addHandlerLast(new ReadTimeoutHandler(props.getTimeoutSeconds(), TimeUnit.SECONDS)));

        return WebClient.builder()
                .baseUrl(props.getBaseUrl())
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .filter(logRequest())
                .filter(logResponse())
                .build();
    }

    private ExchangeFilterFunction logRequest() {
        return ExchangeFilterFunction.ofRequestProcessor(req -> {
            log.debug("LLM API Request: {} {}", req.method(), req.url());
            return Mono.just(req);
        });
    }

    private ExchangeFilterFunction logResponse() {
        return ExchangeFilterFunction.ofResponseProcessor(res -> {
            log.debug("LLM API Response: {}", res.statusCode());
            return Mono.just(res);
        });
    }
}
ENDOFFILE

echo "  ✓ ai-service/src/main/java/com/demo/ai/config/WebClientConfig.java"
mkdir -p "$(dirname "ai-service/src/main/java/com/demo/ai/dto/LlmRequest.java")"
cat > 'ai-service/src/main/java/com/demo/ai/dto/LlmRequest.java' << 'ENDOFFILE'
package com.demo.ai.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

// ─────────────────────────────────────────────────────────────────────────────
// Inbound: what the caller sends us
// ─────────────────────────────────────────────────────────────────────────────

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LlmRequest {

    @NotBlank(message = "prompt must not be blank")
    @Size(max = 32_000, message = "prompt must not exceed 32,000 characters")
    private String prompt;

    /** Optional system / instruction context */
    private String systemPrompt;

    /** Conversation history for multi-turn chat */
    private List<ChatMessage> history;

    /** Max tokens to generate (provider default if null) */
    @Builder.Default
    private Integer maxTokens = 1024;

    /** Sampling temperature 0-2 */
    @Builder.Default
    private Double temperature = 0.7;

    /** Extra provider-specific parameters (e.g. topP, stopSequences) */
    private Map<String, Object> extraParams;

    // ── Nested chat message ──────────────────────────────────────────────────

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ChatMessage {
        private String role;    // "user" or "assistant"
        private String content;
    }
}
ENDOFFILE

echo "  ✓ ai-service/src/main/java/com/demo/ai/dto/LlmRequest.java"
mkdir -p "$(dirname "ai-service/src/main/java/com/demo/ai/dto/LlmResponse.java")"
cat > 'ai-service/src/main/java/com/demo/ai/dto/LlmResponse.java' << 'ENDOFFILE'
package com.demo.ai.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LlmResponse {

    private String text;             // Generated text
    private String provider;         // e.g. "openai", "anthropic", "bedrock"
    private String model;            // e.g. "gpt-4o", "claude-3-5-sonnet"
    private int promptTokens;
    private int completionTokens;
    private int totalTokens;
    private long latencyMs;
    private String finishReason;     // "stop", "length", "content_filter", etc.
}
ENDOFFILE

echo "  ✓ ai-service/src/main/java/com/demo/ai/dto/LlmResponse.java"
mkdir -p "$(dirname "ai-service/src/main/java/com/demo/ai/service/GenericLlmService.java")"
cat > 'ai-service/src/main/java/com/demo/ai/service/GenericLlmService.java' << 'ENDOFFILE'
package com.demo.ai.service;

import com.demo.ai.config.LlmProperties;
import com.demo.ai.dto.LlmRequest;
import com.demo.ai.dto.LlmResponse;
import com.demo.common.exception.ServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Provider-agnostic LLM service.
 *
 * Routing logic:
 *   openai    → POST /chat/completions  (OpenAI format)
 *   anthropic → POST /v1/messages       (Anthropic format)
 *   custom    → POST /chat/completions  (assumes OpenAI-compatible endpoint)
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class GenericLlmService {

    private final WebClient llmWebClient;
    private final LlmProperties props;

    public LlmResponse complete(LlmRequest request) {
        log.info("LLM call | provider={} | model={} | tokens={}",
                props.getProvider(), props.getModel(), request.getMaxTokens());

        long start = System.currentTimeMillis();

        try {
            return switch (props.getProvider().toLowerCase()) {
                case "anthropic" -> callAnthropic(request, start);
                case "openai", "custom" -> callOpenAi(request, start);
                default -> throw ServiceException.badRequest(
                        "Unsupported LLM provider: " + props.getProvider());
            };
        } catch (WebClientResponseException ex) {
            log.error("LLM API error: {} {}", ex.getStatusCode(), ex.getResponseBodyAsString());
            throw ServiceException.internalError(
                    "LLM provider returned error: " + ex.getStatusCode(), ex);
        }
    }

    // ── OpenAI / OpenAI-compatible ───────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private LlmResponse callOpenAi(LlmRequest request, long start) {
        List<Map<String, String>> messages = buildOpenAiMessages(request);

        Map<String, Object> body = new HashMap<>();
        body.put("model", props.getModel());
        body.put("messages", messages);
        body.put("max_tokens", request.getMaxTokens() != null
                ? request.getMaxTokens() : props.getDefaultMaxTokens());
        body.put("temperature", request.getTemperature() != null
                ? request.getTemperature() : props.getDefaultTemperature());

        if (request.getExtraParams() != null) {
            body.putAll(request.getExtraParams());
        }

        Map<String, Object> response = llmWebClient.post()
                .uri("/chat/completions")
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + props.getApiKey())
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .block();

        // Parse OpenAI response
        List<Map<String, Object>> choices =
                (List<Map<String, Object>>) response.get("choices");
        Map<String, Object> choice = choices.get(0);
        Map<String, String> message = (Map<String, String>) choice.get("message");
        String text = message.get("content");
        String finishReason = (String) choice.get("finish_reason");

        Map<String, Object> usage = (Map<String, Object>) response.get("usage");
        int promptTokens = usage != null ? (int) usage.get("prompt_tokens") : 0;
        int completionTokens = usage != null ? (int) usage.get("completion_tokens") : 0;

        return LlmResponse.builder()
                .text(text)
                .provider("openai")
                .model(props.getModel())
                .promptTokens(promptTokens)
                .completionTokens(completionTokens)
                .totalTokens(promptTokens + completionTokens)
                .finishReason(finishReason)
                .latencyMs(System.currentTimeMillis() - start)
                .build();
    }

    // ── Anthropic ────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private LlmResponse callAnthropic(LlmRequest request, long start) {
        List<Map<String, String>> messages = buildOpenAiMessages(request);
        // Anthropic does not include system in messages array
        messages.removeIf(m -> "system".equals(m.get("role")));

        Map<String, Object> body = new HashMap<>();
        body.put("model", props.getModel());
        body.put("messages", messages);
        body.put("max_tokens", request.getMaxTokens() != null
                ? request.getMaxTokens() : props.getDefaultMaxTokens());

        if (request.getSystemPrompt() != null && !request.getSystemPrompt().isBlank()) {
            body.put("system", request.getSystemPrompt());
        }

        Map<String, Object> response = llmWebClient.post()
                .uri("/v1/messages")
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + props.getApiKey())
                .header("x-api-key", props.getApiKey())
                .header("anthropic-version", props.getAnthropicVersion())
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .block();

        List<Map<String, Object>> content =
                (List<Map<String, Object>>) response.get("content");
        String text = (String) content.get(0).get("text");
        String stopReason = (String) response.get("stop_reason");

        Map<String, Object> usage = (Map<String, Object>) response.get("usage");
        int inputTokens  = usage != null ? (int) usage.get("input_tokens") : 0;
        int outputTokens = usage != null ? (int) usage.get("output_tokens") : 0;

        return LlmResponse.builder()
                .text(text)
                .provider("anthropic")
                .model(props.getModel())
                .promptTokens(inputTokens)
                .completionTokens(outputTokens)
                .totalTokens(inputTokens + outputTokens)
                .finishReason(stopReason)
                .latencyMs(System.currentTimeMillis() - start)
                .build();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private List<Map<String, String>> buildOpenAiMessages(LlmRequest request) {
        List<Map<String, String>> messages = new ArrayList<>();

        if (request.getSystemPrompt() != null && !request.getSystemPrompt().isBlank()) {
            messages.add(Map.of("role", "system", "content", request.getSystemPrompt()));
        }

        if (request.getHistory() != null) {
            for (LlmRequest.ChatMessage h : request.getHistory()) {
                messages.add(Map.of("role", h.getRole(), "content", h.getContent()));
            }
        }

        messages.add(Map.of("role", "user", "content", request.getPrompt()));
        return messages;
    }
}
ENDOFFILE

echo "  ✓ ai-service/src/main/java/com/demo/ai/service/GenericLlmService.java"
mkdir -p "$(dirname "ai-service/src/main/java/com/demo/ai/controller/AiController.java")"
cat > 'ai-service/src/main/java/com/demo/ai/controller/AiController.java' << 'ENDOFFILE'
package com.demo.ai.controller;

import com.demo.ai.dto.LlmRequest;
import com.demo.ai.dto.LlmResponse;
import com.demo.ai.service.GenericLlmService;
import com.demo.common.dto.ApiResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/ai")
@RequiredArgsConstructor
@Tag(name = "AI Service", description = "Generic LLM integration — OpenAI, Anthropic, or any compatible API")
public class AiController {

    private final GenericLlmService llmService;

    /**
     * Generic completion — pass any prompt, optional system prompt, optional history.
     */
    @PostMapping("/complete")
    @Operation(summary = "Send a prompt to the configured LLM and get a completion")
    public ResponseEntity<ApiResponse<LlmResponse>> complete(
            @Valid @RequestBody LlmRequest request) {
        LlmResponse response = llmService.complete(request);
        return ResponseEntity.ok(ApiResponse.ok(response));
    }

    /**
     * Summarise text — wraps the LLM with a summarisation system prompt.
     */
    @PostMapping("/summarize")
    @Operation(summary = "Summarise a block of text")
    public ResponseEntity<ApiResponse<LlmResponse>> summarize(
            @RequestBody Map<String, String> body) {
        String text = body.getOrDefault("text", "");

        LlmRequest request = LlmRequest.builder()
                .systemPrompt("You are a professional summariser. Produce a concise, "
                        + "accurate summary in 3-5 sentences. Preserve key facts.")
                .prompt("Please summarise the following text:\n\n" + text)
                .maxTokens(512)
                .temperature(0.3)
                .build();

        return ResponseEntity.ok(ApiResponse.ok(llmService.complete(request)));
    }

    /**
     * Classify text into provided categories.
     */
    @PostMapping("/classify")
    @Operation(summary = "Classify text into one of the provided categories")
    public ResponseEntity<ApiResponse<LlmResponse>> classify(
            @RequestBody Map<String, Object> body) {
        String text       = (String) body.getOrDefault("text", "");
        Object categories = body.getOrDefault("categories", "positive, neutral, negative");

        LlmRequest request = LlmRequest.builder()
                .systemPrompt("You are a text classifier. "
                        + "Reply with ONLY the category name — no explanation.")
                .prompt(String.format(
                        "Classify the following text into one of: [%s].\n\nText: %s",
                        categories, text))
                .maxTokens(64)
                .temperature(0.0)
                .build();

        return ResponseEntity.ok(ApiResponse.ok(llmService.complete(request)));
    }

    /**
     * Health check — returns current provider and model config.
     */
    @GetMapping("/info")
    @Operation(summary = "Return current LLM provider configuration")
    public ResponseEntity<ApiResponse<Map<String, String>>> info(
            com.demo.ai.config.LlmProperties props) {
        return ResponseEntity.ok(ApiResponse.ok(Map.of(
                "provider", props.getProvider(),
                "model",    props.getModel(),
                "baseUrl",  props.getBaseUrl()
        )));
    }
}
ENDOFFILE

echo "  ✓ ai-service/src/main/java/com/demo/ai/controller/AiController.java"
mkdir -p "$(dirname "ai-service/src/main/resources/application.yml")"
cat > 'ai-service/src/main/resources/application.yml' << 'ENDOFFILE'
server:
  port: 8084

spring:
  application:
    name: ai-service

  # API key fetched from Secrets Manager at startup
  # Secret: /myapp/ai-service → { "llm-api-key": "sk-..." }
  config:
    import: "optional:aws-secretsmanager:/myapp/ai-service"

# ── LLM Provider Configuration ────────────────────────────────────────────────
# Switch provider by changing llm.provider — no code changes needed.
#
# To use OpenAI:
#   llm.provider=openai
#   llm.model=gpt-4o
#   llm.base-url=https://api.openai.com/v1
#
# To use Anthropic:
#   llm.provider=anthropic
#   llm.model=claude-3-5-sonnet-20241022
#   llm.base-url=https://api.anthropic.com
#
# To use Ollama locally:
#   llm.provider=custom
#   llm.model=llama3
#   llm.base-url=http://localhost:11434/v1
# ─────────────────────────────────────────────────────────────────────────────
llm:
  provider: ${LLM_PROVIDER:openai}
  model: ${LLM_MODEL:gpt-4o}
  base-url: ${LLM_BASE_URL:https://api.openai.com/v1}
  api-key: ${LLM_API_KEY:}              # injected from Secrets Manager or env var
  default-max-tokens: 1024
  default-temperature: 0.7
  timeout-seconds: 60
  anthropic-version: "2023-06-01"

aws:
  region: ${AWS_REGION:us-east-1}

spring.cloud.aws:
  region:
    static: ${AWS_REGION:us-east-1}
  credentials:
    use-default-aws-credentials-chain: true

management:
  endpoints:
    web:
      exposure:
        include: health, info, metrics

springdoc:
  api-docs:
    path: /api-docs
  swagger-ui:
    path: /swagger-ui.html

logging:
  level:
    com.demo.ai: DEBUG
ENDOFFILE

echo "  ✓ ai-service/src/main/resources/application.yml"
echo ""
echo "✅ All files written successfully!"
echo ""
# ── Step 3: Initialize git ────────────────────────────────────────────────────
echo "── Initialising git repository ──"
git init
git add .
git commit -m "feat: initial commit — Spring Boot 3 microservices with AWS + Docker + AI/LLM

Services:
- api-gateway    (Spring Cloud Gateway, port 8080)
- user-service   (DynamoDB CRUD, port 8081)
- document-service (S3 upload/download + SQS events, port 8082)
- notification-service (SQS listener, port 8083)
- ai-service     (Generic LLM: OpenAI/Anthropic/Bedrock/Ollama, port 8084)

AWS: S3, SQS, DynamoDB, Secrets Manager
Local dev: LocalStack via docker-compose
CI/CD: GitHub Actions -> ECR -> ECS"
echo "✅ Git commit created"
# ── Step 4: Push to GitHub ────────────────────────────────────────────────────
echo ""
echo "── Pushing to GitHub ──"
echo ""
echo "You will be asked for:"
echo "  Username: maharshiec-commits"
echo "  Password: your Personal Access Token (NOT your GitHub password)"
echo ""
git remote add origin https://github.com/maharshiec-commits/springboot3-microservices-docker-ai.git
git branch -M main
git push -u origin main
