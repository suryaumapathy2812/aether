package media

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
	"github.com/google/uuid"
)

type Service struct {
	bucket         string
	bucketTemplate string
	client         *s3.Client
	presign        *s3.PresignClient
	putTTL         time.Duration
	getTTL         time.Duration
	forcePath      bool
	publicBase     string
	defaultCDN     bool
}

type PutURL struct {
	ObjectKey string
	UploadURL string
	Headers   map[string]string
	ExpiresAt time.Time
}

type ObjectInfo struct {
	Key         string
	Size        int64
	ContentType string
	ETag        string
}

func NewFromEnv(ctx context.Context) (*Service, error) {
	bucket := strings.TrimSpace(os.Getenv("S3_BUCKET"))
	bucketTemplate := strings.TrimSpace(os.Getenv("S3_BUCKET_TEMPLATE"))
	if bucket == "" && bucketTemplate == "" {
		return nil, nil
	}
	region := strings.TrimSpace(os.Getenv("S3_REGION"))
	if region == "" {
		region = "us-east-1"
	}

	accessKey := strings.TrimSpace(os.Getenv("S3_ACCESS_KEY_ID"))
	secretKey := strings.TrimSpace(os.Getenv("S3_SECRET_ACCESS_KEY"))
	endpoint := strings.TrimSpace(os.Getenv("S3_ENDPOINT"))
	publicBase := strings.TrimRight(strings.TrimSpace(os.Getenv("S3_PUBLIC_BASE_URL")), "/")

	putTTL := durationFromEnv("S3_PUT_URL_TTL_SECONDS", 300*time.Second)
	getTTL := durationFromEnv("S3_GET_URL_TTL_SECONDS", 900*time.Second)
	forcePath := strings.EqualFold(strings.TrimSpace(os.Getenv("S3_FORCE_PATH_STYLE")), "true")

	loadOpts := []func(*config.LoadOptions) error{config.WithRegion(region)}
	if accessKey != "" && secretKey != "" {
		loadOpts = append(loadOpts, config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(accessKey, secretKey, "")))
	}
	if endpoint != "" {
		resolver := s3.EndpointResolverFromURL(endpoint)
		loadOpts = append(loadOpts, config.WithEndpointResolverWithOptions(aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...any) (aws.Endpoint, error) {
			if service == s3.ServiceID {
				return resolver.ResolveEndpoint(region, s3.EndpointResolverOptions{})
			}
			return aws.Endpoint{}, &aws.EndpointNotFoundError{}
		})))
	}
	cfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return nil, err
	}
	client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.UsePathStyle = forcePath
	})
	presign := s3.NewPresignClient(client)

	return &Service{
		bucket:         bucket,
		bucketTemplate: bucketTemplate,
		client:         client,
		presign:        presign,
		putTTL:         putTTL,
		getTTL:         getTTL,
		forcePath:      forcePath,
		publicBase:     publicBase,
		defaultCDN:     publicBase != "",
	}, nil
}

func (s *Service) Enabled() bool {
	if s == nil || s.client == nil {
		return false
	}
	return strings.TrimSpace(s.bucket) != "" || strings.TrimSpace(s.bucketTemplate) != ""
}

func (s *Service) BucketForUser(userID string) string {
	if s == nil {
		return ""
	}
	if strings.TrimSpace(s.bucketTemplate) == "" {
		return strings.TrimSpace(s.bucket)
	}
	user := sanitizeBucketSegment(userID)
	if user == "" {
		user = "default"
	}
	tmpl := s.bucketTemplate
	if strings.Contains(tmpl, "{user}") {
		return normalizeBucketName(strings.ReplaceAll(tmpl, "{user}", user))
	}
	return normalizeBucketName(tmpl + "-" + user)
}

func (s *Service) BuildObjectKey(userID, sessionID, fileName string) string {
	ext := strings.ToLower(strings.TrimSpace(filepath.Ext(fileName)))
	if len(ext) > 12 {
		ext = ""
	}
	session := sanitizeSegment(sessionID)
	if session == "" {
		session = "chat"
	}
	now := time.Now().UTC()
	if strings.TrimSpace(s.bucketTemplate) != "" {
		return fmt.Sprintf("sessions/%s/%04d/%02d/%02d/%s%s", session, now.Year(), now.Month(), now.Day(), uuid.NewString(), ext)
	}
	user := sanitizeSegment(userID)
	if user == "" {
		user = "default"
	}
	return fmt.Sprintf("users/%s/sessions/%s/%04d/%02d/%02d/%s%s", user, session, now.Year(), now.Month(), now.Day(), uuid.NewString(), ext)
}

func (s *Service) PresignUpload(ctx context.Context, bucket, objectKey, contentType string) (PutURL, error) {
	if !s.Enabled() {
		return PutURL{}, errors.New("media storage is not configured")
	}
	if strings.TrimSpace(objectKey) == "" {
		return PutURL{}, errors.New("object key is required")
	}
	in := &s3.PutObjectInput{
		Bucket: aws.String(strings.TrimSpace(bucket)),
		Key:    aws.String(objectKey),
	}
	if strings.TrimSpace(contentType) != "" {
		in.ContentType = aws.String(contentType)
	}
	res, err := s.presign.PresignPutObject(ctx, in, s3.WithPresignExpires(s.putTTL))
	if err != nil {
		return PutURL{}, err
	}
	headers := map[string]string{}
	for k, vals := range res.SignedHeader {
		if len(vals) > 0 {
			headers[strings.ToLower(k)] = vals[0]
		}
	}
	if strings.TrimSpace(contentType) != "" {
		headers["content-type"] = contentType
	}
	return PutURL{
		ObjectKey: objectKey,
		UploadURL: res.URL,
		Headers:   headers,
		ExpiresAt: time.Now().UTC().Add(s.putTTL),
	}, nil
}

func (s *Service) EnsureBucket(ctx context.Context, bucket string) error {
	bucket = strings.TrimSpace(bucket)
	if bucket == "" {
		return errors.New("bucket is required")
	}
	_, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: aws.String(bucket)})
	if err == nil {
		return nil
	}
	_, err = s.client.CreateBucket(ctx, &s3.CreateBucketInput{Bucket: aws.String(bucket)})
	if err == nil {
		return nil
	}
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		code := strings.TrimSpace(apiErr.ErrorCode())
		if code == "BucketAlreadyOwnedByYou" || code == "BucketAlreadyExists" {
			return nil
		}
	}
	return err
}

func (s *Service) HeadObject(ctx context.Context, bucket, objectKey string) (ObjectInfo, error) {
	out, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{Bucket: aws.String(strings.TrimSpace(bucket)), Key: aws.String(objectKey)})
	if err != nil {
		return ObjectInfo{}, err
	}
	info := ObjectInfo{Key: objectKey}
	if out.ContentLength != nil {
		info.Size = *out.ContentLength
	}
	if out.ContentType != nil {
		info.ContentType = strings.TrimSpace(*out.ContentType)
	}
	if out.ETag != nil {
		info.ETag = strings.Trim(*out.ETag, "\"")
	}
	return info, nil
}

func (s *Service) PresignGet(ctx context.Context, bucket, objectKey string) (string, error) {
	if !s.Enabled() {
		return "", errors.New("media storage is not configured")
	}
	if strings.TrimSpace(objectKey) == "" {
		return "", errors.New("object key is required")
	}
	bucket = strings.TrimSpace(bucket)
	if s.defaultCDN {
		return s.publicBase + "/" + strings.TrimPrefix(objectKey, "/"), nil
	}
	res, err := s.presign.PresignGetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(bucket), Key: aws.String(objectKey)}, s3.WithPresignExpires(s.getTTL))
	if err != nil {
		return "", err
	}
	return res.URL, nil
}

func (s *Service) GetObjectBytes(ctx context.Context, bucket, objectKey string) ([]byte, string, error) {
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(strings.TrimSpace(bucket)), Key: aws.String(objectKey)})
	if err != nil {
		return nil, "", err
	}
	defer out.Body.Close()
	b, err := io.ReadAll(out.Body)
	if err != nil {
		return nil, "", err
	}
	ct := ""
	if out.ContentType != nil {
		ct = strings.TrimSpace(*out.ContentType)
	}
	return b, ct, nil
}

func ExtensionFromMime(m string) string {
	m = strings.TrimSpace(strings.ToLower(m))
	if m == "" {
		return ""
	}
	exts, _ := mime.ExtensionsByType(m)
	if len(exts) == 0 {
		return ""
	}
	return strings.ToLower(exts[0])
}

func sanitizeSegment(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" {
		return ""
	}
	out := make([]rune, 0, len(v))
	for _, r := range v {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return ""
	}
	return string(out)
}

func sanitizeBucketSegment(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" {
		return ""
	}
	out := make([]rune, 0, len(v))
	for _, r := range v {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return ""
	}
	return string(out)
}

func normalizeBucketName(v string) string {
	v = strings.Trim(strings.ToLower(strings.TrimSpace(v)), "-.")
	if len(v) < 3 {
		v = v + "-media"
	}
	if len(v) > 63 {
		v = v[:63]
		v = strings.Trim(v, "-.")
	}
	return v
}

func durationFromEnv(name string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallback
	}
	return time.Duration(n) * time.Second
}

func IsNotFound(err error) bool {
	var nsk *types.NotFound
	return errors.As(err, &nsk)
}
